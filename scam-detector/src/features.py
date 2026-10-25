"""Feature engineering: TF-IDF (word + char), handcrafted scaler."""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler as StandardScaler  # use non-negative scaling for NB compatibility

from .config import (
    CHAR_MIN_DF,
    CHAR_NGRAM_RANGE,
    MAX_FEATURES,
    MODELS_DIR,
    WORD_MIN_DF,
    WORD_NGRAM_RANGE,
    get_logger,
    set_seed,
)

logger = get_logger(__name__)
set_seed()

FEATURES_DIR = MODELS_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


class FeatureBuilder:
    """Build combined features (word_tfidf + char_tfidf + handcrafted)."""

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = WORD_NGRAM_RANGE,
        word_min_df: int = WORD_MIN_DF,
        char_ngram_range: Tuple[int, int] = CHAR_NGRAM_RANGE,
        char_min_df: int = CHAR_MIN_DF,
        max_features: int = MAX_FEATURES,
        scaler_path: Path = FEATURES_DIR / "handcrafted_scaler.joblib",
        word_vec_path: Path = FEATURES_DIR / "word_tfidf.joblib",
        char_vec_path: Path = FEATURES_DIR / "char_tfidf.joblib",
    ) -> None:
        self.word_vec = TfidfVectorizer(
            ngram_range=word_ngram_range,
            min_df=word_min_df,
            max_features=max_features,
            lowercase=False,  # text already cleaned/lowered in preprocess
            token_pattern=r"(?u)\b\w+\b",
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            min_df=char_min_df,
            max_features=max_features // 2,
            lowercase=False,
        )
        self.scaler = StandardScaler(feature_range=(0.0, 1.0))
        self.scaler_path = scaler_path
        self.word_vec_path = word_vec_path
        self.char_vec_path = char_vec_path
        self.handcrafted_cols: List[str] = []

    def fit_transform(
        self,
        text: pd.Series,
        handcrafted_df: pd.DataFrame,
    ) -> np.ndarray:
        X_w = self.word_vec.fit_transform(text.values)
        joblib.dump(self.word_vec, self.word_vec_path)
        logger.info(f"Saved word TF-IDF to {self.word_vec_path} (shape={X_w.shape})")

        X_c = self.char_vec.fit_transform(text.values)
        joblib.dump(self.char_vec, self.char_vec_path)
        logger.info(f"Saved char TF-IDF to {self.char_vec_path} (shape={X_c.shape})")

        # Scale handcrafted features
        self.handcrafted_cols = list(handcrafted_df.columns)
        X_h = self.scaler.fit_transform(handcrafted_df.values)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Saved handcrafted scaler to {self.scaler_path}")
        return self._combine(X_w, X_c, X_h)

    def transform(
        self,
        text: pd.Series,
        handcrafted_df: pd.DataFrame,
    ) -> np.ndarray:
        X_w = self.word_vec.transform(text.values)
        X_c = self.char_vec.transform(text.values)
        X_h = self.scaler.transform(handcrafted_df.values)
        return self._combine(X_w, X_c, X_h)

    @staticmethod
    def _combine(X_w, X_c, X_h) -> np.ndarray:
        from scipy.sparse import hstack

        if hasattr(X_w, "tocsr"):
            X_w = X_w.tocsr()
        if hasattr(X_c, "tocsr"):
            X_c = X_c.tocsr()
        return hstack([X_w, X_c, X_h]).tocsr()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocess import preprocess_dataframe
    from src.data_loader import merge_datasets
    df, _ = merge_datasets()
    dfp = preprocess_dataframe(df)
    builder = FeatureBuilder()
    from sklearn.model_selection import train_test_split
    res = train_test_split(
        dfp["message_clean"],
        dfp[["msg_len","word_count","url_count","digit_ratio","upper_ratio","exclam_count","urgency_count","currency_symbol","has_shortener","suspicious_tld","avg_word_len","special_char_ratio","has_phone","has_url"]],
        dfp["label"],
        test_size=0.2,
        stratify=dfp["label"],
        random_state=42,
    )
    X_tr_text, X_te_text, X_tr_h, X_te_h, y_tr, y_te = res
    X = builder.fit_transform(X_tr_text, X_tr_h)
    print("train combined shape:", X.shape)
