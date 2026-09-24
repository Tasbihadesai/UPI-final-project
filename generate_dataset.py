import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# PARAMETERS
N = 30000
USERS = [f"user_{i}" for i in range(1, 1001)]
MERCHANTS = [f"merchant_{i}" for i in range(1, 1001)]
TXN_TYPES = ["send", "receive", "withdraw", "upi_collect"]
APPS = ["GPay", "PhonePe", "Paytm", "AmazonPay", "BHIM", "Other"]

random.seed(42)
np.random.seed(42)

def generate_row(i):
    """Generate one synthetic, realistic UPI transaction."""
    user = random.choice(USERS)
    merchant = random.choice(MERCHANTS)
    txn_type = random.choice(TXN_TYPES)
    timestamp = datetime(2025, 10, 1) + timedelta(seconds=random.randint(0, 3600 * 24 * 30))
    hour = timestamp.hour
    day_of_week = timestamp.weekday()

    # --- Core attributes ---
    amount = round(np.clip(np.random.exponential(1500), 10, 30000), 2)
    user_avg_amount = round(amount * np.random.uniform(0.6, 1.4), 2)
    user_txn_count = np.random.randint(1, 400)
    user_age_days = np.random.randint(10, 2500)
    user_device_count = np.random.randint(1, 10)
    merchant_risk_score = round(np.clip(np.random.normal(0.5, 0.25), 0, 1), 2)
    ip_change = np.random.choice([0, 1], p=[0.93, 0.07])
    device_change = np.random.choice([0, 1], p=[0.9, 0.1])
    txn_velocity = round(np.random.exponential(2), 2)
    previous_fraud_count = np.random.choice([0, 1, 2, 3], p=[0.9, 0.07, 0.02, 0.01])
    is_new_merchant = np.random.choice([0, 1], p=[0.75, 0.25])
    geo_distance_km = round(np.random.randint(1, 1000), 2)
    upi_app = random.choice(APPS)

    # ------------------------------------------------------------
    # ENHANCED RISK MODEL
    # ------------------------------------------------------------
    risk_score = 0.0

    # 1️⃣ Transaction-based risk
    if amount > 25000: risk_score += 3.0
    elif amount > 15000: risk_score += 2.0
    elif amount > 8000: risk_score += 1.0

    # Transaction type: outgoing (send/withdraw) is riskier
    if txn_type in ["send", "withdraw"]:
        risk_score += 2.0
    elif txn_type == "receive":
        risk_score += 0.2

    # 2️⃣ Merchant risk
    if merchant_risk_score > 0.8:
        risk_score += 2.5
    elif merchant_risk_score > 0.6:
        risk_score += 1.5
    elif merchant_risk_score < 0.3:
        risk_score -= 0.5  # reward for trusted merchants

    # 3️⃣ Device/IP anomalies
    if ip_change and device_change:
        risk_score += 3.0
    elif ip_change or device_change:
        risk_score += 1.5

    # 4️⃣ Temporal patterns
    if hour < 5 or hour > 23:
        risk_score += 1.2
    elif hour < 8:
        risk_score += 0.5
    if day_of_week in [5, 6]:  # weekend
        risk_score += 0.4

    # 5️⃣ Geographical inconsistency
    if geo_distance_km > 800:
        risk_score += 3.0
    elif geo_distance_km > 400:
        risk_score += 1.5
    elif geo_distance_km > 100:
        risk_score += 0.8

    # 6️⃣ Behavioral / User features
    if user_txn_count < 10:
        risk_score += 2.0
    elif user_txn_count < 50:
        risk_score += 0.8
    if user_age_days < 90:
        risk_score += 2.0
    if user_device_count > 5:
        risk_score += 1.2
    if previous_fraud_count > 0:
        risk_score += previous_fraud_count * 1.5

    # 7️⃣ New merchant correlation
    if is_new_merchant and amount > 10000:
        risk_score += 2.0
    elif is_new_merchant:
        risk_score += 0.5

    # 8️⃣ UPI app trust level
    if upi_app == "Other":
        risk_score += 2.0
    elif upi_app in ["GPay", "PhonePe", "Paytm"]:
        risk_score -= 0.3

    # 9️⃣ Transaction velocity
    if txn_velocity > 10:
        risk_score += 1.5
    elif txn_velocity > 5:
        risk_score += 0.8

    # 🔟 Correlation boosts (realistic risk amplification)
    if (ip_change or device_change) and amount > 10000:
        risk_score += 1.2
    if (merchant_risk_score > 0.7) and (geo_distance_km > 400):
        risk_score += 1.0
    if (is_new_merchant and txn_type in ["send", "withdraw"]):
        risk_score += 1.0

    # Safe-pattern decay (reduce false positives)
    if (user_txn_count > 100 and merchant_risk_score < 0.4 and not ip_change):
        risk_score -= 1.0
    if (amount < 500 and txn_type == "receive"):
        risk_score -= 1.0

    # small noise
    risk_score += np.random.normal(0, 0.2)

    # Fraud if risk ≥ 90% equivalent (≈ risk_score ≥ 6.5 empirically)
    prob_fraud = 1 / (1 + np.exp(-1.1 * (risk_score - 3.5)))
    is_fraud = int(prob_fraud >= 0.90)

    return [
        i + 1, timestamp, user, merchant, txn_type, upi_app, amount,
        user_avg_amount, user_txn_count, user_age_days, user_device_count,
        merchant_risk_score, hour, day_of_week, ip_change, device_change,
        txn_velocity, previous_fraud_count, is_new_merchant, geo_distance_km, is_fraud
    ]

# ------------------------------------------------------------
# GENERATE DATASET
# ------------------------------------------------------------
rows = [generate_row(i) for i in range(N)]

cols = [
    "transaction_id", "timestamp", "user_id", "merchant_id", "txn_type", "upi_app", "amount",
    "user_avg_amount", "user_txn_count", "user_age_days", "user_device_count",
    "merchant_risk_score", "hour", "day_of_week", "ip_change", "device_change",
    "txn_velocity", "previous_fraud_count", "is_new_merchant", "geo_distance_km", "is_fraud"
]

df = pd.DataFrame(rows, columns=cols)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
os.makedirs("data", exist_ok=True)
file_path = "data/upi_transactions.csv"
df.to_csv(file_path, index=False)

fraud_rate = df["is_fraud"].mean() * 100
print(f"✅ Generated {len(df)} transactions with {len(df.columns)-1} features + label")
print(f"📊 Fraud rate: {fraud_rate:.2f}%")
print(f"💾 Saved to {os.path.abspath(file_path)}")
