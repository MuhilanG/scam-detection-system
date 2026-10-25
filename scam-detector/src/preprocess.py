"""Text preprocessing and raw-text handcrafted feature extraction."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pandas as pd
import tldextract

from .config import (
    NUM_TOKEN,
    PHONE_TOKEN,
    SUSPICIOUS_TLDS,
    URL_SHORTENERS,
    URL_TOKEN,
    URGENCY_WORDS,
    get_logger,
    set_seed,
)

logger = get_logger(__name__)
set_seed()

# URL detection
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|[a-z0-9.\-]+\.[a-z]{2,}(?:/\S*)?)",
    re.IGNORECASE,
)
PHONE_RE = re.compile(
    r"(\+?\d[\d\-\s()]{7,}\d)",
    re.IGNORECASE,
)
# Common patterns
DIGIT_SEQ_RE = re.compile(r"\d+")
UPPERCASE_RE = re.compile(r"[A-Z]")
EXCLAM_RE = re.compile(r"!")
SPECIAL_RE = re.compile(r"[^A-Za-z0-9\s]+")
MULTISPACE_RE = re.compile(r"\s+")

# Currency symbols
CURRENCY_RE = re.compile(r"[£$₹]")

# Shortener domains
SHORTENER_PATTERN = re.compile(
    r"|".join(re.escape(s) for s in URL_SHORTENERS),
    re.IGNORECASE,
)


def extract_urls(text: str) -> List[str]:
    """Return list of candidate URLs found in text."""
    text = str(text)
    return URL_RE.findall(text)


def has_suspicious_tld(text: str, suspicious: Tuple[str, ...] = SUSPICIOUS_TLDS) -> int:
    """Return 1 if any extracted URL has a suspicious TLD."""
    urls = extract_urls(text)
    for u in urls:
        try:
            ext = tldextract.extract(u)
            tld = "." + ext.suffix if ext.suffix else ""
            if tld.lower() in tuple(t.lower() for t in suspicious):
                return 1
        except Exception:
            continue
    return 0


def has_shortener(text: str) -> int:
    return 1 if SHORTENER_PATTERN.search(str(text)) else 0


def replace_entities(text: str) -> str:
    """Replace URLs, phones and digit sequences with placeholders.

    The ordering is important: replace URLs first (can contain digits),
    then phone numbers, then standalone digit sequences.
    """
    t = str(text)

    # URLs
    t = URL_RE.sub(URL_TOKEN + " ", t)

    # Phones
    t = PHONE_RE.sub(PHONE_TOKEN + " ", t)

    # Digit sequences (but be careful not to affect placeholders - they have no digits; we can do a safe replace)
    # Replace sequences of digits
    t = DIGIT_SEQ_RE.sub(NUM_TOKEN + " ", t)

    # Collapse whitespace
    t = MULTISPACE_RE.sub(" ", t)
    return t.strip().lower()


def handcrafted_features(text: str) -> Dict[str, float]:
    """Compute raw-text handcrafted features (no TF-IDF)."""
    t = str(text)
    urls = extract_urls(t)
    # Count exclamation marks
    exclam_count = len(EXCLAM_RE.findall(t))

    # Uppercase ratio
    upper_matches = len(UPPERCASE_RE.findall(t))
    total_alpha = len(re.findall(r"[A-Za-z]", t)) or 1
    upper_ratio = upper_matches / total_alpha

    # Digit ratio
    digit_matches = len(DIGIT_SEQ_RE.findall(t))
    total_chars = len(t) or 1
    digit_ratio = digit_matches / total_chars

    # Special char ratio
    special_matches = len(SPECIAL_RE.findall(t))
    special_char_ratio = special_matches / total_chars

    # Word stats
    words = t.split()
    word_count = len(words) or 1
    avg_word_len = (sum(len(w) for w in words) / word_count) if words else 0.0

    # Urgency
    urgency_count = 0
    t_lower = t.lower()
    for w in URGENCY_WORDS:
        # simple word-boundary-ish count via regex
        urgency_count += len(re.findall(r"\b" + re.escape(w) + r"\b", t_lower))

    feats = {
        "msg_len": float(len(t)),
        "word_count": float(word_count),
        "url_count": float(len(urls)),
        "digit_ratio": float(digit_ratio),
        "upper_ratio": float(upper_ratio),
        "exclam_count": float(exclam_count),
        "urgency_count": float(urgency_count),
        "currency_symbol": 1.0 if CURRENCY_RE.search(t) else 0.0,
        "has_shortener": float(has_shortener(t)),
        "suspicious_tld": float(has_suspicious_tld(t)),
        "avg_word_len": float(avg_word_len),
        "special_char_ratio": float(special_char_ratio),
        "has_phone": float(1.0 if PHONE_RE.search(t) else 0.0),
        "has_url": float(1.0 if urls else 0.0),
    }
    return feats


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "message",
                         keep_raw: bool = True) -> pd.DataFrame:
    """Return DataFrame with cleaned text and handcrafted features."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in dataframe")
    out = df.copy()
    out["message_raw"] = out[text_col].astype(str)
    out["message_clean"] = out["message_raw"].apply(replace_entities)
    feats = out["message_raw"].apply(handcrafted_features)
    feats_df = pd.json_normalize(feats)
    out = pd.concat([out, feats_df], axis=1)
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import merge_datasets
    df, _ = merge_datasets()
    dfp = preprocess_dataframe(df)
    print(dfp.shape)
    print(dfp[["message_raw", "message_clean"]].head(2).to_string())
    print(dfp[["msg_len", "url_count", "digit_ratio", "urgency_count"]].head(2).to_string())
