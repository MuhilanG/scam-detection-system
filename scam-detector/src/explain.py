"""Explainability using LIME and coefficient/feature importance analysis.

Produces: global feature importance (top 20 scam words, top 20 legit words)
based on LogReg coefficients for word-level contributions; also provides
LIME explanation helper for single messages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from .config import (
    FIGURES_DIR,
    MODEL_COMPARISON_CSV,
    REPORTS_DIR,
    FEATURE_IMPORTANCE_CSV,
    MODELS_DIR,
    get_logger,
    set_seed,
)

FEATURES_DIR = MODELS_DIR / "features"
from .data_loader import merge_datasets
from .features import FeatureBuilder
from .preprocess import preprocess_dataframe

logger = get_logger(__name__)
set_seed()

BUNDLE_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model_bundle.joblib"
COEF_PLOT_SCAM = FIGURES_DIR / "top20_scam_words.png"
COEF_PLOT_LEGIT = FIGURES_DIR / "top20_legit_words.png"
LOCAL_EXPL_PATH = FIGURES_DIR / "lime_explanation.png"


def _get_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Bundle not found: {BUNDLE_PATH}")
    return joblib.load(BUNDLE_PATH)


def analyze_coefficients() -> pd.DataFrame:
    """Fit a reference LogisticRegression on train split to get global feature importances."""
    bundle = _get_bundle()
    from sklearn.model_selection import train_test_split

    df, _ = merge_datasets()
    dfp = preprocess_dataframe(df)
    y = dfp["label"].values
    text = dfp["message_clean"]
    hcols = bundle.get("handcrafted_cols") or [
        "msg_len","word_count","url_count","digit_ratio","upper_ratio","exclam_count","urgency_count",
        "currency_symbol","has_shortener","suspicious_tld","avg_word_len","special_char_ratio","has_phone","has_url",
    ]
    h = dfp[hcols]
    res = train_test_split(text, h, y, test_size=0.2, stratify=y, random_state=42)
    text_tr, text_te, h_tr, h_te, y_tr, y_te = res

    # load vectorizers from features dir
    wv_path = FEATURES_DIR / "word_tfidf.joblib"
    cv_path = FEATURES_DIR / "char_tfidf.joblib"
    sv_path = FEATURES_DIR / "handcrafted_scaler.joblib"
    if not all(p.exists() for p in [wv_path, cv_path, sv_path]):
        builder = FeatureBuilder()
        X_tr = builder.fit_transform(text_tr, h_tr)
    else:
        wv = joblib.load(wv_path)
        cv = joblib.load(cv_path)
        sv = joblib.load(sv_path)
        from scipy.sparse import hstack
        X_w = wv.transform(text_tr)
        X_c = cv.transform(text_tr)
        X_h = sv.transform(h_tr)
        if hasattr(X_w, "tocsr"):
            X_w = X_w.tocsr()
        if hasattr(X_c, "tocsr"):
            X_c = X_c.tocsr()
        X_tr = hstack([X_w, X_c, X_h]).tocsr()

    logreg = LogisticRegression(max_iter=5000, class_weight="balanced", solver="saga")
    logreg.fit(X_tr, y_tr)

    # get word feature names
    wv = joblib.load(wv_path) if wv_path.exists() else None
    word_names = wv.get_feature_names_out().tolist() if wv is not None else []
    # char + handcrafted
    cv = joblib.load(cv_path)
    char_names = cv.get_feature_names_out().tolist()
    h_names = hcols

    # build full feature names
    feature_names = word_names + char_names + h_names
    coefs = logreg.coef_.ravel()

    dfc = pd.DataFrame({"feature": feature_names, "coef": coefs})
    dfc["abs_coef"] = dfc["coef"].abs()
    dfc = dfc.sort_values(by="abs_coef", ascending=False).reset_index(drop=True)

    # split word/char/h for plotting (priority: handcrafted < char < word)
    dfc["type"] = "other"
    dfc.loc[dfc["feature"].isin(h_names), "type"] = "handcrafted"
    dfc.loc[dfc["feature"].isin(char_names), "type"] = "char"
    dfc.loc[dfc["feature"].isin(word_names), "type"] = "word"

    # save
    dfc.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    logger.info(f"Saved feature coefficients to {FEATURE_IMPORTANCE_CSV}")

    # plot top 20 scam words (positive coef -> push to 1/scam) and top 20 legit (negative)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_scam = dfc[(dfc["type"] == "word") & (dfc["coef"] > 0)].head(20)
    top_legit = dfc[(dfc["type"] == "word") & (dfc["coef"] < 0)].sort_values(by="coef", ascending=True).head(20)

    plt.figure(figsize=(6, 5))
    sns.barplot(data=top_scam, x="coef", y="feature", palette="Reds_r")
    plt.title("Top 20 Word Coeffs (Scam-driving)")
    plt.tight_layout()
    plt.savefig(COEF_PLOT_SCAM, dpi=160)
    plt.close()
    logger.info(f"Saved {COEF_PLOT_SCAM}")

    plt.figure(figsize=(6, 5))
    sns.barplot(data=top_legit, x="coef", y="feature", palette="Greens_r")
    plt.title("Top 20 Word Coeffs (Legit-driving)")
    plt.tight_layout()
    plt.savefig(COEF_PLOT_LEGIT, dpi=160)
    plt.close()
    logger.info(f"Saved {COEF_PLOT_LEGIT}")

    return dfc


def lime_explain_single(message: str, num_features: int = 10) -> Dict[str, Any]:
    """Return LIME-style local explanation for a single message."""
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception as e:
        logger.warning(f"LIME not available: {e}")
        return {"error": str(e), "explanation": []}

    bundle = _get_bundle()
    model = bundle["model"]
    builder: FeatureBuilder = bundle["feature_builder"]
    # need text transformer + handcrafted; but LIME works on text; easier to build combined via transform
    from .preprocess import preprocess_dataframe, handcrafted_features, replace_entities

    mdf = pd.DataFrame({"message": [message]})
    mdf = preprocess_dataframe(mdf)
    X = builder.transform(mdf["message_clean"], mdf[bundle["handcrafted_cols"]])

    class Pipe:
        def predict_proba(self, texts):
            # texts is list of cleaned texts from LIME; rebuild
            tmp = pd.DataFrame({"message": texts})
            tmp = preprocess_dataframe(tmp)
            Xt = builder.transform(tmp["message_clean"], tmp[bundle["handcrafted_cols"]])
            return model.predict_proba(Xt)

    explainer = LimeTextExplainer(class_names=["LEGIT", "SCAM"], split_expression=r"\W+")
    exp = explainer.explain_instance(
        replace_entities(message),
        Pipe().predict_proba,
        num_features=num_features,
    )
    # return simple list
    items = []
    for f, w in exp.as_list():
        items.append({"feature": f, "weight": float(w)})
    return {"explanation": items}


if __name__ == "__main__":
    dfc = analyze_coefficients()
    print(dfc.head(5).to_string())
    # quick local example
    res = lime_explain_single("URGENT: Your KYC has expired. Update now at bit.ly/kyc-scam to avoid account block!")
    print(res)
