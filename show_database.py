"""Run: python show_database.py
Print a readable summary of the project's file-based data layer.

This project does not use a traditional SQL/NoSQL database. Instead, it stores
training data in CSV files, model artifacts in .joblib files, and config in
JSON. The script makes that data layer easier to inspect and reason about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent / "scam-detector"


def sep(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def human_size(path: Path) -> str:
    size_kb = path.stat().st_size / 1024
    if size_kb >= 1024:
        return f"{size_kb / 1024:.2f} MB"
    return f"{size_kb:.1f} KB"


def safe_read_csv(path: Path, **kwargs) -> Tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        return pd.read_csv(path, **kwargs), None
    except Exception as exc:  # pragma: no cover - diagnostics only
        return None, f"Could not read {path}: {exc}"


def print_dataset_summary(label: str, df: pd.DataFrame | None, *, head_count: int = 3) -> None:
    if df is None:
        print("Dataset unavailable.")
        return

    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")
    print(f"Null values: {df.isna().sum().to_dict()}")
    print(f"Dtypes: { {k: str(v) for k, v in df.dtypes.items()} }")
    print(df.head(head_count).to_string(index=False))


def print_value_counts(series: pd.Series | None, title: str) -> None:
    print(f"{title}")
    if series is None:
        print("No data available.")
        return
    print(series.value_counts().to_string())


def print_artifacts(folder: Path, label: str) -> None:
    sep(label)
    files = sorted(folder.rglob("*"))
    if not files:
        print(f"No files found in {folder}")
        return

    for file in files:
        if file.is_file():
            print(f"  {file.relative_to(ROOT)}  ({human_size(file)})")


def load_threshold(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"status": "error", "message": str(exc)}


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(f"Project root not found: {ROOT}")

    sep("PROJECT ROOT")
    print(f"Root: {ROOT}")

    spam_path = ROOT / "data/raw/spam.csv"
    custom_path = ROOT / "data/raw/custom_india.csv"
    merged_path = ROOT / "data/processed/merged.csv"
    threshold_path = ROOT / "models/threshold.json"

    spam_df, spam_error = safe_read_csv(spam_path, encoding="latin-1")
    sep("TIER 1: RAW DATA - spam.csv (Kaggle SMS Spam Collection)")
    if spam_error:
        print(spam_error)
    else:
        print_dataset_summary("Kaggle SMS Spam Collection", spam_df)

    custom_df, custom_error = safe_read_csv(custom_path)
    sep("TIER 1: RAW DATA - custom_india.csv (India scam dataset)")
    if custom_error:
        print(custom_error)
    else:
        print_dataset_summary("India scam dataset", custom_df)

    merged_df, merged_error = safe_read_csv(merged_path)
    sep("TIER 2: PROCESSED DATA - merged.csv (unified training set)")
    if merged_error:
        print(merged_error)
    else:
        print_dataset_summary("Merged training set", merged_df)

    if merged_df is not None:
        sep("CLASS BALANCE (label: 0=safe, 1=scam)")
        print(f"Scam rate: {100 * merged_df['label'].mean():.1f}%")
        print(merged_df["label"].value_counts().to_string())

        print_value_counts(merged_df.get("category"), "SCAM CATEGORIES")
        print_value_counts(merged_df.get("source"), "DATA LINEAGE (source)")

    sep("TIER 3: CONFIG - threshold.json")
    print(json.dumps(load_threshold(threshold_path), indent=2))

    print_artifacts(ROOT / "models", "TIER 3: ARTIFACTS (model registry)")
    print_artifacts(ROOT / "reports", "REPORT STORE")

    print("\nSummary: this project uses a file-based data layer rather than a traditional database.")
    print("Data lives in CSV files, model objects live in joblib files, and config lives in JSON.")


if __name__ == "__main__":
    main()
