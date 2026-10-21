#!/usr/bin/env python3
"""FastAPI service for the fraud model. Run: uvicorn api:app --port 8000
Endpoints: GET /health · GET /metrics · POST /score {features...}"""
import json
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

bundle = joblib.load("model.joblib")
pipe, FEATURES, THRESHOLD = bundle["pipe"], bundle["features"], bundle["threshold"]
metrics = json.load(open("metrics.json"))

app = FastAPI(title="Fraud Transaction Detector API")

class Tx(BaseModel):
    amount: float = 100.0
    hour: float = 14.0
    is_weekend: float = 0.0
    tx_count_1h: float = 1.0
    tx_sum_1h: float = 100.0
    amount_vs_avg_ratio: float = 1.0
    distance_from_home_km: float = 5.0
    foreign_merchant: float = 0.0
    card_present: float = 1.0
    merchant_risk: float = 0.2
    account_age_days: float = 365.0
    failed_attempts_24h: float = 0.0

@app.get("/health")
def health():
    return {"ok": True, "threshold": THRESHOLD}

@app.get("/metrics")
def get_metrics():
    return metrics

@app.post("/score")
def score(tx: Tx):
    row = {**tx.dict()}
    row["log_amount"] = float(np.log1p(row["amount"]))
    row["is_night"] = float(row["hour"] < 6 or row["hour"] >= 23)
    X = np.array([[row[f] for f in FEATURES]])
    p = float(pipe.predict_proba(X)[0, 1])
    return {"fraud_probability": round(p, 4), "flagged": p >= THRESHOLD, "threshold": THRESHOLD}
