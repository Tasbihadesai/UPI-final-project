"""
simulator.py
Simple streaming transaction generator for local demo.
Produces dicts with: transaction_id, user_id, amount, timestamp, merchant_id, txn_type
"""

import random
import time
import uuid
from datetime import datetime, timedelta
import numpy as np

USERS = [f"user_{i}" for i in range(1, 101)]
MERCHANTS = [f"merchant_{i}" for i in range(1, 51)]
TXN_TYPES = ["send", "receive", "withdraw", "upi_collect"]

def random_amount():
    # skewed distribution: many small txns, few large
    return round(float(np.random.exponential(scale=200)), 2)

def one_transaction(fraud_rate=0.02):
    t = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": random.choice(USERS),
        "amount": max(1.0, random_amount()),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "merchant_id": random.choice(MERCHANTS),
        "txn_type": random.choice(TXN_TYPES),
    }
    # optionally inject synthetic fraud patterns (for demo)
    if random.random() < fraud_rate:
        # make fraud more likely: high amount + odd merchant
        t["amount"] = round(max(t["amount"], random.uniform(2000, 10000)), 2)
        t["merchant_id"] = "merchant_suspicious"
        t["synthetic_fraud"] = True
    else:
        t["synthetic_fraud"] = False
    return t

def stream_transactions(interval_seconds=0.8, fraud_rate=0.02):
    """
    Generator yields one transaction dict every `interval_seconds`.
    """
    while True:
        yield one_transaction(fraud_rate=fraud_rate)
        time.sleep(interval_seconds)
