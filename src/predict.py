"""
predict.py
Helper functions for loading the saved model and making predictions on new transactions.
"""

import os
import joblib
import pandas as pd
import numpy as np

MODEL_IN = os.path.join(
    os.path.dirname(__file__),
    '..',
    'models',
    'lightgbm_model.pkl'
)


# ---------------------------------------------------------
# DEFAULTS FOR MISSING OPTIONAL FIELDS
# ---------------------------------------------------------
# Live transactions often arrive with only user_id + amount.
# Letting those fields fall through to fillna(0) in predict_transaction()
# was the root cause of every transaction reading as fraud:
#   amount_dev_from_avg = amount - user_avg_amount
# with user_avg_amount missing -> NaN -> 0, so the "deviation" feature
# was silently becoming the full transaction amount every time.
#
# These are NEUTRAL fallbacks, not real user history. If you have real
# stored per-user stats (DB/CSV), look those up in api_server.py and
# pass them in the payload BEFORE calling predict_transaction() --
# these defaults should only kick in for genuinely unknown/new users.
DEFAULT_CONTEXT = {
    "user_txn_count": 10,        # neutral, not "brand new user" (0)
    "user_age_days": 180,        # neutral account age, not "new" (0)
    "user_device_count": 1,
    "merchant_risk_score": 0.1,  # low-but-nonzero baseline
    "txn_velocity": 1,
    "previous_fraud_count": 0,
    "geo_distance_km": 0,
    "ip_change": 0,
    "device_change": 0,
}


def apply_defaults(transaction: dict) -> dict:
    """
    Fills in neutral defaults for any optional field missing from the
    incoming payload, so downstream NaN-handling in preprocess.py never
    has to silently invent a value that skews the model's input.
    Does not mutate the caller's dict.
    """
    txn = dict(transaction)

    for key, default in DEFAULT_CONTEXT.items():
        if key not in txn or txn[key] is None:
            txn[key] = default

    # Special case: if we don't know the user's real average spend,
    # treat THIS transaction as the average -> amount_dev_from_avg = 0
    # (neutral), instead of defaulting to 0 -> amount_dev_from_avg = amount
    # (which reads as an extreme deviation on every single transaction).
    if "user_avg_amount" not in transaction or transaction.get("user_avg_amount") is None:
        txn["user_avg_amount"] = txn["amount"]

    return txn


def load_model():
    data = joblib.load(MODEL_IN)
    return (
        data['model'],
        data.get('encoders', {}),
        data.get('feature_names', [])
    )


def prepare_single(transaction: dict, encoders: dict):
    df = pd.DataFrame([transaction])

    from preprocess import basic_clean, feature_engineer

    df = basic_clean(df)
    df, _ = feature_engineer(df)

    return df


def predict_transaction(transaction: dict):

    transaction = apply_defaults(transaction)

    model, encoders, feature_names = load_model()

    df = prepare_single(transaction, encoders)


    # APPLY SAME ENCODING AS TRAINING
    for col, le in encoders.items():

        if col in df.columns:

            df[col] = df[col].astype(str)

            df[col] = df[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_
                else -1
            )


    # ALIGN FEATURES WITH TRAINING
    for col in feature_names:

        if col not in df.columns:
            df[col] = 0


    # Keep same feature order
    df = df[feature_names]


    X = df.fillna(0)


    # GET FRAUD PROBABILITY
    try:
        proba = model.predict_proba(X)[0][1]

    except AttributeError:
        proba = model.predict(X)[0]


    # FRAUD DECISION
    label = int(proba >= 0.5)


    return {
        "probability_fraud": float(proba),
        "is_fraud": label
    }



if __name__ == "__main__":

    # Case 1: full payload (as before) — should still flag as high risk,
    # since these values are genuinely fraud-like.
    sample_txn_full = {

        "user_id": "user_999",

        "merchant_id": "unknown",

        "txn_type": "send",

        "UPI_app": "gpay",

        "amount": 95000,

        "user_avg_amount": 5000,

        "user_txn_count": 25,

        "user_age_days": 5,

        "user_device_count": 3,

        "merchant_risk_score": 0.95,

        "ip_change": 1,

        "device_change": 1,

        "txn_velocity": 20,

        "previous_fraud_count": 5,

        "is_new_merchant": 1,

        "geo_distance_km": 250

    }

    print("Full payload result:", predict_transaction(sample_txn_full))

    # Case 2: minimal payload (only user_id + amount) — this is the
    # scenario that was previously always coming back as fraud.
    sample_txn_minimal = {
        "user_id": "user_123",
        "amount": 500,
    }

    print("Minimal payload result:", predict_transaction(sample_txn_minimal))