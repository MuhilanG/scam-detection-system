"""Central configuration: paths, seed and hyperparameters.

All modules import paths and settings from here so that a single edit
propagates through the whole pipeline.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

SPAM_CSV: Path = RAW_DIR / "spam.csv"           # Kaggle SMS Spam Collection
CUSTOM_CSV: Path = RAW_DIR / "custom_india.csv"  # Indian-context synthetic set
MERGED_CSV: Path = PROCESSED_DIR / "merged.csv"

MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODEL_COMPARISON_CSV: Path = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_JSON: Path = MODELS_DIR / "threshold.json"
FEATURE_IMPORTANCE_CSV: Path = REPORTS_DIR / "feature_importance.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy (and torch if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a configured logger writing to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file:
        log_path = ROOT / "logs"
        log_path.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_path / log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Preprocessing / feature settings
# --------------------------------------------------------------------------- #
URL_TOKEN: str = "<url>"
PHONE_TOKEN: str = "<phone>"
NUM_TOKEN: str = "<num>"

# Shorteners frequently abused by phishers.
URL_SHORTENERS: Tuple[str, ...] = (
    "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd",
    "cutt.ly", "rb.gy", "shorturl", "buff.ly", "s.id", "urlshortener",
)

# TLDs with a bad reputation for phishing campaigns.
SUSPICIOUS_TLDS: Tuple[str, ...] = (".xyz", ".top", ".click", ".zip", ".mov",
                                    ".icu", ".cfd", ".rest", ".buzz", ".tk",
                                    ".ml", ".ga", ".cf", ".gq", ".live")

URGENCY_WORDS: Tuple[str, ...] = (
    "urgent", "immediately", "instantly", "hurry", "limited", "expire",
    "expires", "expired", "last", "final", "act", "now", "fast", "quick",
    "within", "hrs", "hours", "mins", "minutes", "deadline", "warning",
    "suspend", "suspended", "block", "blocked", "terminate", "suspend",
    "verify", "verifynow", "donotdelay", "asap", "restricted", "locked",
)

# --------------------------------------------------------------------------- #
# Training settings
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.15          # carved out of the *train* split for threshold tuning
TARGET_PRECISION: float = 0.985  # pick threshold reaching ~98-99% precision
N_FOLDS: int = 5
LABELS: Tuple[str, ...] = ("LEGIT", "SCAM")

# TF-IDF
WORD_NGRAM_RANGE: Tuple[int, int] = (1, 2)
WORD_MIN_DF: int = 2
CHAR_NGRAM_RANGE: Tuple[int, int] = (3, 5)
CHAR_MIN_DF: int = 3
MAX_FEATURES: int = 60_000

# Handcrafted (raw-text) features
HANDCRAFTED_FEATURES: List[str] = field(default_factory=list)  # populated below


HANDCRAFTED_FEATURES = [
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


@dataclass(frozen=True)
class ModelSpec:
    """A candidate classifier used by :mod:`src.train`."""

    name: str
    factory: str  # key handled inside train.get_models()


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("MultinomialNB", "mnb"),
    ModelSpec("LogReg", "logreg"),
    ModelSpec("LinearSVM", "svm"),
    ModelSpec("RandomForest", "rf"),
    ModelSpec("XGBoost", "xgb"),
)

DEFAULT_FEATURE_CONFIG: Dict[str, object] = {
    "word_ngram_range": WORD_NGRAM_RANGE,
    "char_ngram_range": CHAR_NGRAM_RANGE,
    "seed": SEED,
}

if __name__ == "__main__":  # pragma: no cover
    for _p in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        _p.mkdir(parents=True, exist_ok=True)
        print(f"ensured {_p}")
    print(f"ROOT = {ROOT}")
