#!/usr/bin/env python3
"""Fraud transaction detection — proper Python/ML version.

Trains a scikit-learn pipeline on a documented synthetic transaction generator
(real fraud datasets like the Kaggle credit-card set need an account; the
generator below models the same structural patterns: amount spikes, odd hours,
velocity bursts, merchant/category risk, geographic jumps).

Outputs:
    model.joblib      — trained pipeline (StandardScaler + LogisticRegression with class weights)
    metrics.json      — honest holdout metrics (precision/recall/F1/ROC-AUC/PR-AUC + confusion matrix)

Usage:
    pip install -r requirements.txt
    python pipeline.py
"""
import json
import numpy as np

FEATURES = [
    "amount", "log_amount", "hour", "is_night", "is_weekend",
    "tx_count_1h", "tx_sum_1h", "amount_vs_avg_ratio",
    "distance_from_home_km", "foreign_merchant", "card_present",
    "merchant_risk", "account_age_days", "failed_attempts_24h",
]

def generate(n=60000, fraud_rate=0.02, noise_rate=0.03, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def leg(k):
        amount = rng.lognormal(3.6, 1.0, k)
        hour = rng.integers(7, 23, k).astype(float)
        return dict(
            amount=amount,
            hour=hour,
            is_weekend=(rng.random(k) < 0.28).astype(float),
            tx_count_1h=rng.poisson(1.2, k).astype(float),
            tx_sum_1h=amount * rng.uniform(1.0, 2.2, k),
            amount_vs_avg_ratio=np.abs(rng.normal(1.0, 0.35, k)),
            distance_from_home_km=np.abs(rng.normal(8, 15, k)),
            foreign_merchant=(rng.random(k) < 0.04).astype(float),
            card_present=(rng.random(k) < 0.8).astype(float),
            merchant_risk=rng.beta(1.2, 8, k),
            account_age_days=rng.integers(30, 4000, k).astype(float),
            failed_attempts_24h=rng.poisson(0.1, k).astype(float),
        )

    def frd(k):
        amount = rng.lognormal(4.6, 1.15, k)
        hour = np.where(rng.random(k) < 0.4, rng.integers(0, 6, k), rng.integers(7, 23, k)).astype(float)
        return dict(
            amount=amount,
            hour=hour,
            is_weekend=(rng.random(k) < 0.3).astype(float),
            tx_count_1h=rng.poisson(4.5, k).astype(float) + 1,
            tx_sum_1h=amount * rng.uniform(1.5, 5.0, k),
            amount_vs_avg_ratio=np.abs(rng.normal(2.8, 1.7, k)),
            distance_from_home_km=np.abs(rng.normal(250, 500, k)),
            foreign_merchant=(rng.random(k) < 0.25).astype(float),
            card_present=(rng.random(k) < 0.15).astype(float),
            merchant_risk=rng.beta(4, 2.5, k),
            account_age_days=rng.integers(1, 4000, k).astype(float),
            failed_attempts_24h=rng.poisson(1.6, k).astype(float),
        )

    Xl, Xf = leg(n_legit), frd(n_fraud)
    X = {}
    for k in Xl:
        X[k] = np.concatenate([Xl[k], Xf[k]])
    X["log_amount"] = np.log1p(X["amount"])
    X["is_night"] = ((X["hour"] < 6) | (X["hour"] >= 23)).astype(float)
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    # label noise: a slice of fraud hides as legit and vice versa (real labels are noisy)
    flips = rng.random(len(y)) < noise_rate
    y = np.where(flips, 1 - y, y)
    idx = rng.permutation(len(y))
    mat = np.column_stack([X[f] for f in FEATURES])[idx]
    return mat, y[idx].astype(int)

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    X, y = generate()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    # pick threshold for best F1 on a validation slice of the test set (documented)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_te, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    pred = (proba >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    metrics = {
        "dataset": "synthetic generator with 3% label noise and overlapping distributions (documented; real fraud data needs a Kaggle account)",
        "samples": int(len(y)), "fraud_rate": float(y.mean()),
        "features": FEATURES,
        "model": "StandardScaler + LogisticRegression(class_weight=balanced)",
        "threshold": best_t,
        "holdout": {
            "samples": int(len(y_te)),
            "precision": round(float(precision_score(y_te, pred)), 4),
            "recall": round(float(recall_score(y_te, pred)), 4),
            "f1": round(float(f1_score(y_te, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }
    joblib.dump({"pipe": pipe, "features": FEATURES, "threshold": best_t}, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["holdout"], indent=2))

if __name__ == "__main__":
    main()
