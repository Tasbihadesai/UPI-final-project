"""
preprocess.py
Data preprocessing and feature engineering for UPI Fraud Detection
(Compatible with 20-feature advanced dataset)
"""

import pandas as pd
import numpy as np


# -------------------------------------------------------
# BASIC CLEANING
# -------------------------------------------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw UPI transaction data:
     - Standardizes column names
     - Converts numeric types
     - Handles missing values
     - Parses timestamps
    """
    df.columns = [c.strip().lower() for c in df.columns]

    # Convert timestamp to datetime if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Numeric conversions
    num_cols = [
        "amount", "user_avg_amount", "user_txn_count", "user_age_days",
        "user_device_count", "merchant_risk_score", "hour", "day_of_week",
        "txn_velocity", "previous_fraud_count", "geo_distance_km"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------
def feature_engineer(df: pd.DataFrame):
    """
    Adds derived features to enhance fraud detection signals.
    Handles merges safely (no duplicate column issues).
    """

    # --- 1️⃣ USER TRANSACTION COUNT (safe join)
    if "user_id" in df.columns and "transaction_id" in df.columns:
        user_counts = (
            df.groupby("user_id")["transaction_id"]
            .count()
            .rename("user_total_txn_count")
        )
        df = df.join(user_counts, on="user_id", how="left", rsuffix="_dup")
    elif "user_txn_count" in df.columns:
        # Live single-transaction predictions have no transaction_id,
        # so fall back to the user-supplied txn count instead of losing this feature to 0
        df["user_total_txn_count"] = df["user_txn_count"]


    # --- 2️⃣ AMOUNT DEVIATION FROM USER AVERAGE ---
    if "amount" in df.columns and "user_avg_amount" in df.columns:
        df["amount_dev_from_avg"] = df["amount"] - df["user_avg_amount"]

    # --- 3️⃣ LOG OF AMOUNT (helps LightGBM learn scale variations) ---
    if "amount" in df.columns:
        df["log_amount"] = np.log1p(df["amount"])

    # --- 4️⃣ TIME-BASED FEATURES (recomputes hour/day if missing) ---
    if "timestamp" in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
        df["day_of_week"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.weekday

    # --- 5️⃣ HIGH-RISK TIME WINDOWS ---
    # Early hours or late-night transactions are often fraud-prone
    if "hour" in df.columns:
        df["is_odd_hour"] = df["hour"].apply(lambda x: 1 if (x < 5 or x > 22) else 0)

    # --- 6️⃣ DEVICE/IP CHANGE INTERACTIONS ---
    if "ip_change" in df.columns and "device_change" in df.columns:
        df["device_ip_combo"] = df["ip_change"].astype(str) + "_" + df["device_change"].astype(str)

    # --- 7️⃣ USER RISK SCORE (synthetic aggregate indicator) ---
    # Higher if user is new, uses many devices, or has past frauds
        # --- 7️⃣ USER RISK SCORE (synthetic aggregate indicator) ---
    # Higher if user is new, uses many devices, or has past frauds
    risk_factors = []
    if "user_age_days" in df.columns:
        # Floor the age at 14 days for this formula only — prevents a
        # brand-new (age=0) account from producing an extreme outlier
        # value the model never saw a realistic range of during training.
        effective_age = df["user_age_days"].clip(lower=14)
        risk_factors.append((1 / (effective_age + 1)) * 1000)
    if "user_device_count" in df.columns:
        risk_factors.append(df["user_device_count"] * 5)
    if "previous_fraud_count" in df.columns:
        risk_factors.append(df["previous_fraud_count"] * 10)

    if len(risk_factors) > 0:
        df["user_behavior_risk"] = np.sum(risk_factors, axis=0)

    # --- 8️⃣ Drop any duplicate columns (safety) ---
    df = df.loc[:, ~df.columns.duplicated()]

    return df, None
