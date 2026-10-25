"""Evaluation utilities: metrics, confusion matrix, PR curve, threshold selection plot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

from .config import FIGURES_DIR, MODELS_DIR, THRESHOLD_JSON, get_logger, set_seed
from .data_loader import merge_datasets
from .features import FeatureBuilder
from .preprocess import preprocess_dataframe

logger = get_logger(__name__)
set_seed()

BUNDLE_PATH = MODELS_DIR / "best_model_bundle.joblib"
CM_PATH = FIGURES_DIR / "confusion_matrix.png"
PR_PATH = FIGURES_DIR / "pr_curve.png"


def load_bundle(bundle_path: Path = BUNDLE_PATH):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}. Run src.train first.")
    return joblib.load(bundle_path)


def generate_evaluation_plots(bundle_path: Path = BUNDLE_PATH) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(bundle_path)

    # reload data to get test set
    from sklearn.model_selection import train_test_split

    df, _ = merge_datasets()
    dfp = preprocess_dataframe(df)
    y = dfp["label"].values
    text = dfp["message_clean"]
    hcols = bundle.get("handcrafted_cols") or [
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
    res = train_test_split(text, h, y, test_size=0.2, stratify=y, random_state=42)
    text_tr, text_te, h_tr, h_te, y_tr, y_te = res

    builder: FeatureBuilder = bundle["feature_builder"]
    X_te = builder.transform(text_te, h_te)
    model = bundle["model"]
    try:
        y_proba = model.predict_proba(X_te)[:, 1]
    except Exception:
        yd = model.decision_function(X_te)
        yd = (yd - yd.min()) / (yd.max() - yd.min() + 1e-8)
        y_proba = yd

    try:
        thr = json.load(open(THRESHOLD_JSON))["threshold"]
    except Exception:
        thr = 0.5
    y_pred = (y_proba >= thr).astype(int)

    # CM
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["LEGIT", "SCAM"], yticklabels=["LEGIT", "SCAM"])
    plt.title(f"Confusion Matrix (thr={thr:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=160)
    plt.close()
    logger.info(f"Saved confusion matrix to {CM_PATH}")

    # PR curve
    precision, recall, _ = precision_recall_curve(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    plt.axvline(x=float(recall[y_pred.sum() > -1]) if False else recall[np.searchsorted(precision, precision[0])])  # no-op guard
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Scam class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PR_PATH, dpi=160)
    plt.close()
    logger.info(f"Saved PR curve to {PR_PATH}")


if __name__ == "__main__":
    generate_evaluation_plots()
    print("evaluation plots written")
