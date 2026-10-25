# Scam Message Detection System

SMS / WhatsApp scam classifier with a Streamlit UI that scores a message's scam
risk and shows **why** it was flagged — highlighted suspicious words, URL
reputation flags, and per-message feature weights.

Built with scikit-learn: word + character TF-IDF combined with 14 handcrafted
features, five candidate models compared by PR-AUC, and a decision threshold
tuned on a validation split for ~98–99% precision.

## Highlights

- **India-aware dataset** — Kaggle SMS Spam Collection (5,574 rows) merged with a
  custom Indian-context set (202 rows) covering UPI refund fraud, KYC/OTP
  phishing, digital arrest scams, fake jobs, lottery, and customs/courier fraud.
- **Hybrid features** — word TF-IDF (1–2 grams), char TF-IDF (3–5 grams), and
  handcrafted signals (URLs, shorteners, suspicious TLDs, urgency words, digit
  ratio, uppercase ratio, phone presence, ...).
- **Model selection** — MultinomialNB, LogReg, LinearSVM, RandomForest, XGBoost
  compared on a held-out test split and 5-fold CV; best model kept by PR-AUC on
  the scam class.
- **Threshold tuning** — tuned on a validation split (never the test set) so
  precision stays at the target while maximising recall.
- **Explainability** — coefficient-based per-token contributions in the UI, plus
  LIME on demand, and global top-20 scam/legit word charts.
- **Interactive app** — verdict card, risk gauge, inline highlights, link
  inspection, and actionable "what to do" guidance.

## Results

Test-set metrics (`reports/model_comparison.csv`):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **LinearSVM** | 0.9597 | 0.9470 | 0.9533 | 0.9944 | **0.9894** |
| LogReg | 0.9463 | 0.9338 | 0.9400 | 0.9953 | 0.9853 |
| RandomForest | 0.9536 | 0.9536 | 0.9536 | 0.9934 | 0.9840 |
| XGBoost | 0.9722 | 0.9272 | 0.9492 | 0.9948 | 0.9828 |
| MultinomialNB | 0.8810 | 0.9801 | 0.9279 | 0.9905 | 0.9788 |

Selected model: **LinearSVM** (calibrated) with decision threshold **0.7229**,
chosen on the validation split for ~98.5% precision (recall ≈ 0.857 at that
threshold). See `scam-detector/models/threshold.json`.

## Project structure

```
scam-detection-system/
├── README.md
├── DATABASE_EXPLANATION.md        # file-based data layer explained
├── show_database.py               # prints a summary of datasets/artifacts
└── scam-detector/
    ├── app/streamlit_app.py       # interactive UI
    ├── src/
    │   ├── config.py              # paths, seed, hyperparameters
    │   ├── data_loader.py         # load + merge + dedupe raw CSVs
    │   ├── preprocess.py          # text cleaning + handcrafted features
    │   ├── features.py            # word/char TF-IDF + scaler
    │   ├── train.py               # model comparison, CV, threshold tuning
    │   ├── evaluate.py            # confusion matrix + PR curve
    │   ├── explain.py             # coefficients + LIME explanations
    │   └── predict.py             # inference API
    ├── data/
    │   ├── raw/                   # spam.csv, custom_india.csv
    │   └── processed/             # merged.csv (generated)
    ├── models/                    # bundles, vectorizers, threshold.json
    ├── reports/                   # comparison CSV + figures
    ├── tests/test_predict.py      # preprocessing + inference tests
    ├── requirements.txt
    └── .streamlit/config.toml     # theme
```

## Quickstart

Python 3.10+ recommended. Run commands from the `scam-detector/` directory.

```bash
cd scam-detector
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Pretrained artifacts are already included, so you can launch the app directly:

```bash
streamlit run app/streamlit_app.py
```

### Retrain from scratch

```bash
python -m src.data_loader   # build data/processed/merged.csv
python -m src.train         # compare models, tune threshold, save bundle
python -m src.evaluate      # confusion matrix + PR curve in reports/figures
python -m src.explain       # feature importances + top-word plots
```

Training writes `models/best_model_bundle.joblib`, `models/features/*.joblib`,
`models/threshold.json`, and `reports/model_comparison.csv`.

### Predict from the CLI

```bash
python -m src.predict "URGENT: your KYC expired, verify at sbi-kyc.top/update within 24 hrs"
```

Prints a JSON result with `label_str`, `probability`, `threshold`, `reasons`,
`urls`, and `handcrafted` features.

### Run tests

```bash
pytest -q
```

### Inspect the data layer

```bash
python show_database.py   # from the repo root
```

## How it works

1. **Load & merge** — both raw CSVs are normalised (`ham`/`spam` → `0`/`1`),
   tagged with `source`/`category`, deduplicated, and written to `merged.csv`.
2. **Clean text** — URLs, phone numbers, and digit sequences are replaced with
   `<url>`, `<phone>`, `<num>` placeholders; text is lowercased.
3. **Build features** — word TF-IDF, char TF-IDF, and scaled handcrafted
   features are horizontally stacked into one sparse matrix (fitted on train only).
4. **Train & compare** — five classifiers are trained on an 80/20 stratified
   split with 5-fold CV; the best PR-AUC model is kept.
5. **Tune threshold** — a 15% slice of the training data is used to pick the
   threshold that reaches the target precision with the highest recall.
6. **Explain & serve** — coefficient × TF-IDF weights produce per-token reasons,
   and the Streamlit app renders the verdict, gauge, highlights, and URL flags.

## Limitations

- The custom Indian dataset is **synthetic** (202 rows); real-world distribution
  may differ.
- Binary SMS-style text classifier — not a replacement for security controls.
- English-focused; Hinglish/regional-language coverage is limited.
- Static pretrained artifacts: retraining is required after data changes.

## Documentation

`DATABASE_EXPLANATION.md` covers the file-based storage design (no SQL/NoSQL),
data schemas, ETL pipeline, artifact registry, and the path to a production
database if scan history or multi-user serving is added.
