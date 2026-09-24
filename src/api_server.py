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
    "lightgbm_model.pkl"
)

LIVE_DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "live_transactions.csv"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = None
encoders = {}

if os.path.exists(MODEL_PATH):
    try:
        data = joblib.load(MODEL_PATH)

        model = data.get("model")
        encoders = data.get("encoders", {})

        print("Model loaded successfully.")

    except Exception as e:
        print("Model loading error:", e)

else:
    print("Model file not found:", MODEL_PATH)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "UPI Fraud API running",
        "model_loaded": model is not None,
        "mode": "transaction_simulation"
    })


# ---------------------------------------------------------
# TRANSACTION
# ---------------------------------------------------------

@app.route("/transaction", methods=["POST"])
def transaction():

    # 1. Receive JSON
    payload = request.get_json()

    if not payload:
        return jsonify({
            "success": False,
            "error": "No JSON received"
        }), 400


    # 2. Validate required fields
    required_fields = [
        "user_id",
        "amount"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in payload
    ]

    if missing_fields:
        return jsonify({
            "success": False,
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400


    # 3. Timestamp
    payload["timestamp"] = datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # 4. Run fraud model
    try:

        result = predict_transaction(payload)

        prob = float(result["probability_fraud"])
        is_fraud = int(result["is_fraud"])

    except Exception as e:

        print("Prediction error:", e)

        # IMPORTANT:
        # Never treat a prediction failure as SAFE.
        return jsonify({
            "success": False,
            "error": "Fraud prediction could not be completed",
            "transaction_status": "NOT_APPROVED",
            "payment_status": "NOT_EXECUTED"
        }), 503


    # 5. Risk decision
    if prob >= 0.50:

        risk_level = "HIGH"
        transaction_status = "BLOCKED"
        payment_status = "NOT_EXECUTED"
        receiver_credited = False

        message = (
            "Transaction blocked because high fraud risk "
            "was detected."
        )

    else:

        risk_level = "LOW"
        transaction_status = "APPROVED"
        payment_status = "SIMULATED_SUCCESS"
        receiver_credited = True

        message = (
            "Transaction approved for simulated payment."
        )


    # 6. Save transaction
    os.makedirs(
        os.path.dirname(LIVE_DATA_PATH),
        exist_ok=True
    )

    transaction_record = {

        "timestamp": payload["timestamp"],

        "user_id": payload.get(
            "user_id",
            "user_anon"
        ),

        "merchant_id": payload.get(
            "merchant_id",
            "unknown"
        ),

        "txn_type": payload.get(
            "txn_type",
            "send"
        ),

        "amount": payload.get(
            "amount",
            0.0
        ),

        "fraud_probability": prob,

        "is_fraud": is_fraud,

        "risk_level": risk_level,

        "transaction_status": transaction_status,

        "payment_status": payment_status,

        "receiver_credited": receiver_credited
    }

    header = not os.path.exists(
        LIVE_DATA_PATH
    )

    pd.DataFrame(
        [transaction_record]
    ).to_csv(
        LIVE_DATA_PATH,
        mode="a",
        header=header,
        index=False
    )


    # 7. Return decision to client
    return jsonify({

        "success": True,

        "fraud_probability": prob,

        "is_fraud": is_fraud,

        "risk_level": risk_level,

        "transaction_status": transaction_status,

        "payment_status": payment_status,

        "receiver_credited": receiver_credited,

        "message": message,

        "simulation": True
    })


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )