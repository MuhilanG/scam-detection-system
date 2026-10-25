"""Dataset loading and preparation.

Loads the Kaggle SMS Spam Collection (v1,v2) and the custom Indian-context
CSV (message,label,category) provided in data/raw/, merges them, normalises
labels to 0/1, deduplicates messages and reports class balance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import (
    CUSTOM_CSV,
    MERGED_CSV,
    PROCESSED_DIR,
    SPAM_CSV,
    get_logger,
    set_seed,
)

logger = get_logger(__name__)

set_seed()


def _clean_text(text: str) -> str:
    return str(text).strip().replace("\r", " ").replace("\n", " ")


def load_spam(base_path: Path = SPAM_CSV) -> pd.DataFrame:
    """Load Kaggle SMS Spam Collection dataset."""
    if not base_path.exists():
        raise FileNotFoundError(f"spam.csv not found at {base_path}")
    df = pd.read_csv(base_path, encoding="latin-1", usecols=["v1", "v2"], dtype=str)
    df = df.rename(columns={"v1": "label_str", "v2": "message"})
    df["message"] = df["message"].astype(str).map(_clean_text)
    df["label"] = df["label_str"].str.lower().map({"ham": 0, "spam": 1}).astype(int)
    df["source"] = "sms_spam"
    df["category"] = "spam_kaggle" if True else "spam_kaggle"
    # populate category consistently
    df.loc[df["label"] == 1, "category"] = "spam_kaggle"
    df.loc[df["label"] == 0, "category"] = "ham_kaggle"
    return df[["message", "label", "category", "source"]].copy()


def load_custom(base_path: Path = CUSTOM_CSV) -> pd.DataFrame:
    """Load Indian-context synthetic CSV."""
    if not base_path.exists():
        raise FileNotFoundError(f"custom_india.csv not found at {base_path}")
    df = pd.read_csv(base_path, encoding="utf-8", dtype={"label": "Int64"})
    df["message"] = df["message"].astype(str).map(_clean_text)
    df["label"] = df["label"].astype("Int64").astype(int)
    df["source"] = "custom_india"
    if "category" not in df.columns:
        df["category"] = "unknown"
    return df[["message", "label", "category", "source"]].copy()


def merge_datasets(spam_df: Optional[pd.DataFrame] = None,
                   custom_df: Optional[pd.DataFrame] = None,
                   dedupe: bool = True) -> Tuple[pd.DataFrame, Path]:
    """Merge datasets, normalise and deduplicate."""
    if spam_df is None:
        spam_df = load_spam()
    if custom_df is None:
        custom_df = load_custom()
    df = pd.concat([spam_df, custom_df], ignore_index=True, sort=False)

    # Drop exact message duplicates
    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["message"], keep="first")
        dropped = before - len(df)
        logger.info(f"Dropped {dropped} duplicate messages")

    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved merged dataset to {MERGED_CSV} (shape={df.shape})")
    return df, MERGED_CSV


if __name__ == "__main__":
    df, out = merge_datasets()
    print(df.shape)
    print(df["label"].value_counts().to_string())
    print(df["source"].value_counts().to_string())
    print(f"Written: {out}")
