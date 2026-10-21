# Python/ML version (scikit-learn + FastAPI)

> Part of **[fraud-transaction-detector](https://github.com/AKASH991833/fraud-transaction-detector)** — live browser demo: https://akash991833.github.io/fraud-transaction-detector/ · screenshots in the repo-root `screenshots/` folder.


The GitHub Pages demo runs a compact pre-trained model in your browser. This
folder is the full training + serving pipeline behind it.

## Run

```bash
pip install -r requirements.txt
python pipeline.py        # trains + evaluates, writes model.joblib + metrics.json
uvicorn api:app --port 8000
```

- `GET /metrics` — honest holdout metrics (precision/recall/F1/ROC-AUC/PR-AUC, confusion matrix, chosen threshold)
- `POST /score` — score one transaction JSON → `fraud_probability` + `flagged`

## Data honesty

Real credit-card fraud datasets (e.g. the Kaggle ULB set) require an account,
so `pipeline.py` documents and uses a synthetic generator modeling the same
structural patterns (amount spikes, night hours, velocity bursts, distance,
merchant risk). Metrics in `metrics.json` are computed on a stratified 20%
holdout and mean exactly what they say — for this synthetic distribution.
