"""Fraud Transaction Detector — Streamlit UI over the trained sklearn pipeline."""
import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "ml-api" / "model.joblib")
pipe, FEATURES, THRESHOLD = bundle["pipe"], bundle["features"], bundle["threshold"]
metrics = json.load(open(ROOT / "ml-api" / "metrics.json"))

st.set_page_config(page_title="Fraud Transaction Detector", page_icon="🛡️")
st.title("🛡️ Fraud Transaction Detector")
st.caption("scikit-learn model served live · [Browser demo](https://akash991833.github.io/fraud-transaction-detector/) · [Code](https://github.com/AKASH991833/fraud-transaction-detector)")

st.subheader("Score a transaction")
c1, c2, c3 = st.columns(3)
with c1:
    amount = st.number_input("Amount (₹/$)", 1.0, 1_000_000.0, 100.0, 1.0)
    hour = st.slider("Hour of day", 0, 23, 14)
    tx_count_1h = st.number_input("Transactions in last hour", 0, 100, 1)
    failed = st.number_input("Failed attempts (24h)", 0, 50, 0)
with c2:
    ratio = st.number_input("Amount vs usual (ratio)", 0.0, 50.0, 1.0, 0.1)
    distance = st.number_input("Distance from home (km)", 0.0, 20000.0, 8.0, 1.0)
    merchant_risk = st.slider("Merchant risk score", 0.0, 1.0, 0.15, 0.01)
    age = st.number_input("Account age (days)", 0, 10000, 400)
with c3:
    weekend = st.checkbox("Weekend")
    foreign = st.checkbox("Foreign merchant")
    card_present = st.checkbox("Card physically present", value=True)
    tx_sum_1h = st.number_input("Spend in last hour", 0.0, 1_000_000.0, float(amount), 1.0)

if st.button("Score", type="primary"):
    row = {
        "amount": amount, "log_amount": float(np.log1p(amount)), "hour": float(hour),
        "is_night": float(hour < 6 or hour >= 23), "is_weekend": float(weekend),
        "tx_count_1h": float(tx_count_1h), "tx_sum_1h": float(tx_sum_1h),
        "amount_vs_avg_ratio": float(ratio), "distance_from_home_km": float(distance),
        "foreign_merchant": float(foreign), "card_present": float(card_present),
        "merchant_risk": float(merchant_risk), "account_age_days": float(age),
        "failed_attempts_24h": float(failed),
    }
    X = np.array([[row[f] for f in FEATURES]])
    p = float(pipe.predict_proba(X)[0, 1])
    if p >= THRESHOLD:
        st.error(f"⚠️ Flagged as suspicious — fraud probability **{p:.1%}** (threshold {THRESHOLD:.0%})")
    else:
        st.success(f"Looks normal — fraud probability **{p:.1%}** (threshold {THRESHOLD:.0%})")
    st.progress(min(p, 1.0))

with st.expander("Model & honest metrics"):
    st.json(metrics)
    st.markdown(
        "Trained on a **documented synthetic generator** (public fraud datasets need accounts), "
        "with label noise so scores stay believable. Real-world performance will differ — "
        "treat flags as a prompt to verify, never proof."
    )
