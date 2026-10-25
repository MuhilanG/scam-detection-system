"""Train, compare and persist models.

Trains candidate classical ML models with same stratified 80/20 split and
5-fold stratified CV. Selects best model by PR-AUC for the scam class (1).
Saves the best pipeline components and a full bundle to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .config import (
    FIGURES_DIR,
    MODELS_DIR,
    MODEL_COMPARISON_CSV,
    N_FOLDS,
    SEED,
    TARGET_PRECISION,
    TEST_SIZE,
    THRESHOLD_JSON,
    get_logger,
    set_seed,
)
from .data_loader import merge_datasets
from .features import FeatureBuilder
from .preprocess import preprocess_dataframe

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

logger = get_logger(__name__)
set_seed(SEED)

BUNDLE_PATH = MODELS_DIR / "best_model_bundle.joblib"


def get_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LogReg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            n_jobs=None,
        ),
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=10_000),
            cv=3,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=None,  # handled by imbalance via threshold/metrics
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def _split_validation(train_idx: np.ndarray, val_size: float = 0.15,
                     y_tr: np.ndarray | pd.Series | list = None) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    if y_tr is None:
        raise ValueError("y_tr required for stratified validation split")
    y_t = np.array(y_tr)
    idx_tr, idx_val = train_test_split(
        train_idx,
        test_size=val_size,
        stratify=y_t,
        random_state=SEED,
    )
    return idx_tr, idx_val


def train_and_compare() -> Dict[str, Any]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df, _ = merge_datasets()
    dfp = preprocess_dataframe(df)

    # Stratified split
    y = dfp["label"].values
    text = dfp["message_clean"]
    hcols = [
        "msg_len",
        "word_count",
        "url_count",
        "digit_ratio",
        "upper_ratio",
        "exclam_count",
        "urgency_count",
        "currency_symbol",
        "has_shortener",
        "suspicious_tld",
        "avg_word_len",
        "special_char_ratio",
        "has_phone",
        "has_url",
    ]
    h = dfp[hcols]

    res = train_test_split(
        text,
        h,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED,
    )
    text_tr, text_te, h_tr, h_te, y_tr, y_te = res

    # Fit feature builder on train only
    builder = FeatureBuilder()
    X_tr = builder.fit_transform(text_tr, h_tr)
    X_te = builder.transform(text_te, h_te)

    # Also carve validation from train for threshold tuning
    idx_all_tr = np.arange(X_tr.shape[0])
    idx_tr_sub, idx_val = train_test_split(
        idx_all_tr,
        test_size=0.15,
        stratify=y_tr,
        random_state=SEED,
    )
    X_tr_sub, y_tr_sub = X_tr[idx_tr_sub], y_tr[idx_tr_sub]
    X_val, y_val = X_tr[idx_val], y_tr[idx_val]

    models = get_models()
    results_rows: List[Dict[str, Any]] = []
    best_name = "MultinomialNB"
    best_pr_auc = -1.0
    best_est = models[best_name]

    for name, est in models.items():
        logger.info(f"Training {name}...")
        est.fit(X_tr_sub, y_tr_sub)
        y_pred = est.predict(X_te)
        try:
            y_proba = est.predict_proba(X_te)[:, 1]
        except Exception:
            # some calibrated models
            try:
                y_proba = est.predict_proba(X_te)[:, 1]
            except Exception:
                y_proba = y_pred.astype(float)

        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        try:
            roc = roc_auc_score(y_te, y_proba)
        except Exception:
            roc = np.nan
        try:
            pr = average_precision_score(y_te, y_proba)
        except Exception:
            pr = np.nan

        row = {
            "model": name,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
        }
        results_rows.append(row)
        if not np.isnan(pr) and pr > best_pr_auc:
            best_pr_auc = pr
            best_est = est
            best_name = name
        logger.info(f"{name}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} ROC={roc:.3f} PR={pr:.3f}")

    # Cross-validation on train set (sub) for robustness
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_results: Dict[str, Dict[str, float]] = {}
    for name, est in models.items():
        try:
            scores = cross_validate(
                est,
                X_tr_sub,
                y_tr_sub,
                cv=cv,
                scoring={
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                    "pr_auc": "average_precision",
                },
                n_jobs=-1,
            )
            cv_results[name] = {
                "precision_mean": float(np.mean(scores["test_precision"])),
                "precision_std": float(np.std(scores["test_precision"])),
                "recall_mean": float(np.mean(scores["test_recall"])),
                "recall_std": float(np.std(scores["test_recall"])),
                "f1_mean": float(np.mean(scores["test_f1"])),
                "f1_std": float(np.std(scores["test_f1"])),
                "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "roc_auc_std": float(np.std(scores["test_roc_auc"])),
                "pr_auc_mean": float(np.mean(scores["test_pr_auc"])),
                "pr_auc_std": float(np.std(scores["test_pr_auc"])),
            }
            logger.info(f"CV {name}: PR={cv_results[name]['pr_auc_mean']:.3f}±{cv_results[name]['pr_auc_std']:.3f}")
        except Exception as e:
            logger.warning(f"CV failed for {name}: {e}")

    df_cmp = pd.DataFrame(results_rows).sort_values(by="pr_auc", ascending=False)
    df_cmp.to_csv(MODEL_COMPARISON_CSV, index=False)
    logger.info(f"Saved model comparison to {MODEL_COMPARISON_CSV}")

    # Threshold tuning on validation set using best estimator (retrain on train_sub to predict val)
    best_est.fit(X_tr_sub, y_tr_sub)
    try:
        yv_proba = best_est.predict_proba(X_val)[:, 1]
    except Exception:
        yv_proba = best_est.decision_function(X_val)
        # min-max approx not needed for threshold search; use raw if sparse-like
        yv_proba = (yv_proba - yv_proba.min()) / (yv_proba.max() - yv_proba.min() + 1e-8)

    thresholds = np.linspace(0.01, 0.99, 500)
    best_t = 0.5
    best_rec_at_t = 0.0
    for t in thresholds:
        yv_pred_t = (yv_proba >= t).astype(int)
        p_t = precision_score(y_val, yv_pred_t, zero_division=0)
        if p_t >= TARGET_PRECISION - 1e-4:
            r_t = recall_score(y_val, yv_pred_t, zero_division=0)
            if r_t >= best_rec_at_t:
                best_rec_at_t = r_t
                best_t = float(t)
            # also can break or continue; prefer higher recall among those meeting precision
    # if none exactly, pick threshold with highest precision >= target or just max precision
    if best_rec_at_t == 0.0:
        # fallback: choose t giving precision closest to target from above or max
        scores = []
        for t in thresholds:
            yv_pred_t = (yv_proba >= t).astype(int)
            p_t = precision_score(y_val, yv_pred_t, zero_division=0)
            r_t = recall_score(y_val, yv_pred_t, zero_division=0)
            scores.append((p_t, r_t, t))
        # pick highest p_t, tie -> highest r_t
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # find first with p_t>=target else top
        cand = None
        for p_t, r_t, t in scores:
            if p_t >= TARGET_PRECISION - 1e-3:
                cand = (p_t, r_t, t)
                break
        if cand is None:
            cand = scores[0] if scores else (0, 0, 0.5)
        _, best_rec_at_t, best_t = cand

    thresh_info = {
        "model": best_name,
        "threshold": float(best_t),
        "target_precision": float(TARGET_PRECISION),
        "recall_at_threshold": float(best_rec_at_t),
        "note": "chosen on validation split (not test set)",
    }
    with open(THRESHOLD_JSON, "w", encoding="utf-8") as f:
        json.dump(thresh_info, f, indent=2)
    logger.info(f"Saved threshold to {THRESHOLD_JSON}: t={best_t:.3f} P>=~{TARGET_PRECISION} R@t={best_rec_at_t:.3f}")

    # Fit best estimator on full train (tr_sub union val) before saving final metrics on test? better: retrain best on all X_tr,y_tr
    best_est.fit(X_tr, y_tr)
    try:
        y_te_proba = best_est.predict_proba(X_te)[:, 1]
    except Exception:
        try:
            yd = best_est.decision_function(X_te)
            yd = (yd - yd.min()) / (yd.max() - yd.min() + 1e-8)
            y_te_proba = yd
        except Exception:
            y_te_proba = best_est.predict(X_te).astype(float)

    y_te_pred_def = (y_te_proba >= best_t).astype(int)
    final = {
        "best_model": best_name,
        "threshold": float(best_t),
        "precision_test": float(precision_score(y_te, y_te_pred_def, zero_division=0)),
        "recall_test": float(recall_score(y_te, y_te_pred_def, zero_division=0)),
        "f1_test": float(f1_score(y_te, y_te_pred_def, zero_division=0)),
        "roc_auc_test": float(roc_auc_score(y_te, y_te_proba)) if len(np.unique(y_te)) > 1 else np.nan,
        "pr_auc_test": float(average_precision_score(y_te, y_te_proba)),
    }
    logger.info(f"FINAL (test@t={best_t:.3f}): P={final['precision_test']:.3f} R={final['recall_test']:.3f} F1={final['f1_test']:.3f} PR={final['pr_auc_test']:.3f}")

    # Save full bundle for inference
    bundle = {
        "feature_builder": builder,
        "model": best_est,
        "threshold": float(best_t),
        "best_model_name": best_name,
        "handcrafted_cols": hcols,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    logger.info(f"Saved best model bundle to {BUNDLE_PATH}")

    return {
        "comparison": df_cmp.to_dict(orient="records"),
        "cv_results": cv_results,
        "threshold": thresh_info,
        "final": final,
        "bundle_path": str(BUNDLE_PATH),
        "model_comparison_csv": str(MODEL_COMPARISON_CSV),
    }


if __name__ == "__main__":
    out = train_and_compare()
    print(json.dumps(out["final"], indent=2))
    print("done")
