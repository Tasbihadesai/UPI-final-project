"""
api_server.py

UPI Fraud Detection API
Receives a transaction, runs the ML fraud model,
makes a simulated transaction decision, and stores the result.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import numpy as np
from datetime import datetime

from predict import predict_transaction


app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "lightgbm_model.pkl",
)

LIVE_DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "live_transactions.csv",
)


# ---------------------------------------------------------
# LOAD MODEL (health check only)
# ---------------------------------------------------------

model = None
encoders = {}

if os.path.exists(MODEL_PATH):
    try:
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict):
            model = data.get("model")
            encoders = data.get("encoders", {})
        else:
            model = data
        print("Model loaded successfully.")
    except Exception as e:
        print("Model loading error:", e)
else:
    print("Model file not found:", MODEL_PATH)


def normalize_probability(value) -> float:
    """Normalize either 0..1 or 0..100 input to decimal 0..1."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid fraud probability returned by model.")

    if not np.isfinite(p):
        raise ValueError("Non-finite fraud probability returned by model.")

    if p > 1.0:
        p /= 100.0

    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "UPI Fraud API running",
        "model_loaded": model is not None,
        "mode": "transaction_simulation",
    })


# ---------------------------------------------------------
# TRANSACTION
# ---------------------------------------------------------

@app.route("/transaction", methods=["POST"])
def transaction():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "success": False,
            "error": "No valid JSON received",
        }), 400

    required_fields = ["user_id", "amount"]
    missing_fields = [
        field for field in required_fields
        if field not in payload or payload.get(field) in (None, "")
    ]

    if missing_fields:
        return jsonify({
            "success": False,
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    # Preserve the transaction_id/timestamp supplied by Streamlit whenever
    # available so the same ID appears in history, API output and live data.
    if not payload.get("transaction_id"):
        payload["transaction_id"] = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        result = predict_transaction(payload)
        prob = normalize_probability(
            result.get("fraud_probability", result.get("probability_fraud"))
        )
        is_fraud = int(result.get("is_fraud", int(prob >= 0.50)))
        decision_threshold = normalize_probability(
            result.get("decision_threshold", 0.50)
        )
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({
            "success": False,
            "error": "Fraud prediction could not be completed",
            "transaction_status": "NOT_APPROVED",
            "payment_status": "NOT_EXECUTED",
            "receiver_credited": False,
        }), 503

    # The model's own threshold determines the block decision. This avoids
    # silently replacing a threshold stored with the trained model.
    is_fraud = int(is_fraud == 1 or prob >= decision_threshold)

    if is_fraud:
        risk_level = "HIGH"
        transaction_status = "BLOCKED"
        payment_status = "NOT_EXECUTED"
        receiver_credited = False
        message = "Transaction blocked because high fraud risk was detected."
    elif prob >= 0.30:
        risk_level = "MEDIUM"
        transaction_status = "APPROVED"
        payment_status = "SIMULATED_SUCCESS"
        receiver_credited = True
        message = "Transaction approved; risk score is being monitored."
    else:
        risk_level = "LOW"
        transaction_status = "APPROVED"
        payment_status = "SIMULATED_SUCCESS"
        receiver_credited = True
        message = "Transaction approved for simulated payment."

    os.makedirs(os.path.dirname(LIVE_DATA_PATH), exist_ok=True)

    transaction_record = {
        "timestamp": payload["timestamp"],
        "transaction_id": payload.get("transaction_id"),
        "user_id": payload.get("user_id", "user_anon"),
        "merchant_id": payload.get("merchant_id", "unknown"),
        "txn_type": payload.get("txn_type", "send"),
        "upi_app": payload.get("upi_app", payload.get("UPI_app", "unknown")),
        "amount": payload.get("amount", 0.0),
        "fraud_probability": prob,
        "is_fraud": is_fraud,
        "decision_threshold": decision_threshold,
        "risk_level": risk_level,
        "transaction_status": transaction_status,
        "payment_status": payment_status,
        "receiver_credited": receiver_credited,
    }

    header = not os.path.exists(LIVE_DATA_PATH)
    pd.DataFrame([transaction_record]).to_csv(
        LIVE_DATA_PATH,
        mode="a",
        header=header,
        index=False,
    )

    return jsonify({
        "success": True,
        "transaction_id": payload.get("transaction_id"),
        "fraud_probability": prob,
        "probability_fraud": prob,
        "is_fraud": is_fraud,
        "decision_threshold": decision_threshold,
        "risk_level": risk_level,
        "transaction_status": transaction_status,
        "payment_status": payment_status,
        "receiver_credited": receiver_credited,
        "message": message,
        "simulation": True,
    })


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
