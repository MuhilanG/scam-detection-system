# Database / Data Storage Layer — Complete Explanation

> **Project:** Scam Detection System (SMS/WhatsApp scam message classifier)
> **Core statement:** This project uses **no traditional database** (no SQL/NoSQL). It uses a **file-based data layer** — CSV files for data, joblib pickles for model artifacts, and JSON for configuration. This is the standard, intentional design pattern for classical ML projects.

---

## 1. Why No Traditional Database?

| Factor | Explanation |
|---|---|
| **Dataset size** | ~5,400 messages, under 1 MB total — a database server is overkill |
| **Access pattern** | Data is read once during training and read-only during inference — no concurrent writes, no transactions |
| **ML convention** | Training data lives in CSV (universally portable); trained models are serialized files |
| **Simplicity & portability** | Zero setup, zero administration, zero cost — clone the repo and run |
| **Reproducibility** | Static files guarantee the exact same training data every time |

**When a real database WOULD be needed:** multi-user scan history, feedback loops, real-time serving at scale, or datasets in the hundreds of thousands of rows.

---

## 2. Three-Tier Storage Architecture

```
┌──────────────────────────────────────────────────────────┐
│  TIER 1 — RAW DATA (immutable source CSVs)               │
│                                                          │
│  data/raw/spam.csv          5,574 rows   0.48 MB         │
│  data/raw/custom_india.csv    202 rows   0.03 MB         │
├──────────────────────────────────────────────────────────┤
│  TIER 2 — PROCESSED DATA (cleaned training set)          │
│                                                          │
│  data/processed/merged.csv  5,362 rows   0.54 MB         │
├──────────────────────────────────────────────────────────┤
│  TIER 3 — ARTIFACTS (models, config, reports)            │
│                                                          │
│  models/best_model_bundle.joblib       2.33 MB           │
│  models/features/*.joblib              1.38 MB           │
│  models/threshold.json                 tiny              │
│  reports/*.csv + figures/*.png         2.2+ MB           │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Data Schemas (Equivalent of Table Definitions)

### Table 1 — `spam.csv` (Kaggle SMS Spam Collection — base dataset)

| Column | Type | Values | Description |
|---|---|---|---|
| `v1` | string | `ham` / `spam` | Original label (ham = legitimate) |
| `v2` | string | free text | The SMS message |

- **Rows:** 5,574
- **Encoding:** latin-1
- **Origin:** UCI Machine Learning Repository / Kaggle public dataset

### Table 2 — `custom_india.csv` (India-specific scam dataset — synthetic)

| Column | Type | Values | Description |
|---|---|---|---|
| `message` | string | free text | Scam/legit message |
| `label` | integer | `0` / `1` | 0 = safe, 1 = scam |
| `category` | string | `upi_refund`, `kyc_otp`, `digital_arrest`, `fake_job`, ... | Scam taxonomy |

- **Rows:** 202
- **Purpose:** Adds Indian-context scams the Kaggle dataset lacks — UPI refund fraud, KYC/OTP phishing, digital arrest scams, fake job offers

### Table 3 — `merged.csv` (Unified training set — the "fact table")

| Column | Type | Description |
|---|---|---|
| `message` | string | Input text |
| `label` | integer | 0 = safe, 1 = scam (harmonized target) |
| `category` | string | Scam category (or `ham_kaggle` for legit) |
| `source` | string | `sms_spam` or `custom_india` — full data lineage/traceability |

- **Rows:** 5,362 (after deduplication from 5,776 combined)
- **Class balance:** ~87% safe / ~13% scam (typical real-world imbalance)

### Config Store — `threshold.json` (model decision configuration)

```json
{
  "model": "LinearSVM",
  "threshold": 0.7229,
  "target_precision": 0.985,
  "recall_at_threshold": 0.8571,
  "note": "chosen on validation split (not test set)"
}
```

This replaces what would be a `config` or `parameters` table in a database.

---

## 4. Data Lifecycle / ETL Pipeline

```
spam.csv + custom_india.csv          (Tier 1: raw)
        │
        ▼
  data_loader.py  ── load, fix encoding, rename columns,
        │            harmonize labels, deduplicate
        ▼
   merged.csv                              (Tier 2: processed)
        │
        ▼
  preprocess.py  ── clean text, mask URLs/phones/digits,
        │            extract 14 handcrafted features
        ▼
  features.py  ── fit word TF-IDF, char TF-IDF, MinMaxScaler
        │
        ▼
  train.py  ── train 5 models, 5-fold CV, tune threshold
        │
        ▼
  best_model_bundle.joblib + threshold.json  (Tier 3: artifacts)
        │
        ▼
  predict.py / Streamlit app  ── read-only inference
```

**ETL steps in detail:**
1. **Extract** — read both raw CSVs (encoding fixed to latin-1)
2. **Transform** — rename columns (`v1`→`label`, `v2`→`message`), map `ham`→0 / `spam`→1, add `category` and `source` columns
3. **Load** — union both tables, remove exact/near duplicates (5,776 → 5,362 rows), write `merged.csv`

---

## 5. Model Artifact Store (File-Based "Model Registry")

| File | Size | Contents |
|---|---|---|
| `best_model_bundle.joblib` | 2.33 MB | Complete inference package: trained LinearSVC + fitted FeatureBuilder + threshold + feature column list |
| `features/word_tfidf.joblib` | 0.40 MB | Fitted word-level TF-IDF vectorizer |
| `features/char_tfidf.joblib` | 0.98 MB | Fitted character n-gram TF-IDF vectorizer |
| `features/handcrafted_scaler.joblib` | ~0 MB | Fitted MinMaxScaler for the 14 handcrafted features |
| `threshold.json` | tiny | Tuned decision threshold (0.7229) |

**Serialization technology: joblib** — Python's standard for persisting scikit-learn objects. It uses pickle with efficient compression, designed specifically for large numpy/scipy matrices (like TF-IDF sparse matrices). A single `.joblib` file loads the entire trained system into memory with one call.

---

## 6. Report / Analytics Store (Audit Trail)

| File | Purpose |
|---|---|
| `reports/model_comparison.csv` | Metrics table for all 5 models (equivalent to an experiments table) |
| `reports/feature_importance.csv` | Coefficient weights for every feature (2.1 MB) |
| `reports/figures/confusion_matrix.png` | Visual error analysis |
| `reports/figures/pr_curve.png` | Precision-recall trade-off |
| `reports/figures/top20_scam_words.png` | Most predictive scam tokens |
| `reports/figures/top20_legit_words.png` | Most predictive legitimate tokens |

---

## 7. Path to Production — Adding a Real Database (Future Scope / Q&A)

| Requirement | Recommended Solution |
|---|---|
| Store user scan history | **PostgreSQL** table: `scans(id, message, verdict, confidence, timestamp)` |
| User feedback / retraining loop | Table: `feedback(scan_id, user_reported_label, corrected_label)` |
| Caching repeated messages | **Redis** key-value cache (message hash → verdict) |
| API backend | **FastAPI** service reading/writing PostgreSQL |
| Large corpora (100k+ rows) | **Apache Parquet** files on cloud storage, or **MongoDB** for unstructured text |
| Experiment tracking | **MLflow** or a `model_runs` table replacing the comparison CSV |

---

## 🎤 Presentation Summary (30-second version)

1. **No traditional database — by design.** Flat-file storage (CSV + joblib + JSON) is the standard for ML projects of this scale.
2. **Three tiers:** Raw CSVs → Processed CSV → Serialized model artifacts.
3. **Two data sources** (Kaggle SMS dataset + custom Indian scam dataset) merged into one unified 5,362-row training set with full source traceability.
4. **Model persistence** via joblib — the entire trained system fits in a single 2.3 MB file.
5. **Scaling path:** SQLite/PostgreSQL + FastAPI + Redis would be added only when multi-user serving and scan history are required.
