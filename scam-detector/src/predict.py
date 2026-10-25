"""Inference API: predict(message) -> {label, probability, reasons, urls}."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from .config import (
    FEATURE_IMPORTANCE_CSV,
    MODELS_DIR,
    THRESHOLD_JSON,
    get_logger,
    set_seed,
)
from .preprocess import extract_urls, handcrafted_features, has_shortener, has_suspicious_tld, preprocess_dataframe, replace_entities

logger = get_logger(__name__)
set_seed()

BUNDLE_PATH = MODELS_DIR / "best_model_bundle.joblib"


class ScamDetector:
    """Load the trained bundle and score messages."""

    def __init__(self, bundle_path: Path = BUNDLE_PATH):
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. Train first with: python -m src.train"
            )
        self.bundle = joblib.load(bundle_path)
        self.builder = self.bundle["feature_builder"]
        self.model = self.bundle["model"]
        self.handcrafted_cols = self.bundle["handcrafted_cols"]
        self.threshold = float(self.bundle.get("threshold", 0.5))

    def predict_proba(self, message: str) -> float:
        mdf = preprocess_dataframe(pd.DataFrame({"message": [message]}))
        X = self.builder.transform(mdf["message_clean"], mdf[self.handcrafted_cols])
        try:
            proba = float(self.model.predict_proba(X)[0, 1])
        except Exception:
            # fallback: decision function scaled to [0,1]
            try:
                s = float(self.model.decision_function(X)[0])
                proba = 1.0 / (1.0 + pow(2.718281828, -s))
            except Exception:
                proba = 1.0 if int(self.model.predict(X)[0]) == 1 else 0.0
        return proba

    def predict(self, message: str) -> Dict[str, Any]:
        proba = self.predict_proba(message)
        label = 1 if proba >= self.threshold else 0
        label_str = "SCAM" if label == 1 else "SAFE"

        # Top contributing features (coefficient-based, if available)
        reasons = self._top_reasons(message)

        # URL analysis
        urls = self._analyze_urls(message)

        # Handcrafted features
        hf = handcrafted_features(message)

        return {
            "message": message,
            "label": label,
            "label_str": label_str,
            "probability": proba,
            "threshold": self.threshold,
            "reasons": reasons,
            "urls": urls,
            "handcrafted": hf,
        }

    def _top_reasons(self, message: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Top contributing words: coef(word) * tfidf(word) from a reference LogReg.

        The reference coefficients come from reports/feature_importance.csv
        (produced by src.explain). Falls back to a keyword heuristic if absent.
        """
        mdf = preprocess_dataframe(pd.DataFrame({"message": [message]}))
        X_w = self.builder.word_vec.transform(mdf["message_clean"].values)

        try:
            coef_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
            word_vec = self.builder.word_vec
            feature_names = word_vec.get_feature_names_out().tolist()
            name_to_coef = dict(
                zip(coef_df.loc[coef_df["type"] == "word", "feature"],
                    coef_df.loc[coef_df["type"] == "word", "coef"])
            )
            # tfidf values for this message
            row = X_w.tocsr()[0]
            contributions: List[Dict[str, Any]] = []
            for col_idx, val in zip(row.indices, row.data):
                w = feature_names[col_idx]
                c = name_to_coef.get(w)
                if c is None or val == 0:
                    continue
                contributions.append({"feature": w, "weight": float(c * val)})
            contributions.sort(key=lambda d: d["weight"], reverse=True)
            if contributions:
                # return strongest scam-side and legit-side signals mixed by abs weight
                contributions.sort(key=lambda d: abs(d["weight"]), reverse=True)
                return contributions[:top_k]
        except Exception as e:
            logger.debug(f"coefficient-based reasoning unavailable: {e}")

        # Fallback heuristic reasons (keyword presence + weights)
        from .config import URGENCY_WORDS
        t = message.lower()
        reasons: List[Dict[str, Any]] = []
        for w in URGENCY_WORDS:
            if re.search(r"\b" + re.escape(w) + r"\b", t):
                reasons.append({"feature": w, "weight": 0.1})
        if has_shortener(message):
            reasons.append({"feature": "url_shortener", "weight": 0.3})
        if has_suspicious_tld(message):
            reasons.append({"feature": "suspicious_tld", "weight": 0.3})
        if extract_urls(message):
            reasons.append({"feature": "has_url", "weight": 0.2})
        reasons.sort(key=lambda d: d["weight"], reverse=True)
        return reasons[:top_k]

    def _analyze_urls(self, message: str) -> List[Dict[str, Any]]:
        urls = extract_urls(message)
        out = []
        for u in urls:
            out.append({
                "url": u,
                "shortener": bool(has_shortener(u)),
                "suspicious_tld": bool(has_suspicious_tld(u)),
            })
        return out


# Singleton for app reuse
_detector: Optional[ScamDetector] = None


def get_detector(bundle_path: Path = BUNDLE_PATH) -> ScamDetector:
    global _detector
    if _detector is None:
        _detector = ScamDetector(bundle_path)
    return _detector


def predict(message: str) -> Dict[str, Any]:
    """Convenience wrapper: predict(message) -> dict result."""
    return get_detector().predict(message)


if __name__ == "__main__":
    import sys

    msg = sys.argv[1] if len(sys.argv) > 1 else (
        "URGENT: Your Rs.5,000 refund is pending. Claim now at flipkart-refund.xyz/claim or your UPI will be blocked!"
    )
    res = predict(msg)
    print(json.dumps(res, indent=2, default=str))
