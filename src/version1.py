import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import time
import os
import csv
from datetime import datetime

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_DB = os.path.join(DATA_DIR, "users.csv")
HISTORY_DIR = os.path.join(DATA_DIR, "user_history")
MODEL_PATH = os.path.join(BASE_DIR, "models", "randomforest_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ------------------------------------------------------------
# HISTORY CSV COLUMNS
# ------------------------------------------------------------
HISTORY_COLUMNS = [
    "timestamp","user_id","merchant_id","txn_type","upi_app","amount",
    "user_avg_amount","user_txn_count","user_age_days","user_device_count",
    "merchant_risk_score","hour","day_of_week","ip_change","device_change",
    "txn_velocity","previous_fraud_count","is_new_merchant","geo_distance_km",
    "fraud_probability","fraud_flag"
]

# ------------------------------------------------------------
# PASSWORD HASHING
# ------------------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------------------------------------------------
# USERS DB HANDLING
# ------------------------------------------------------------
def ensure_users_db():
    if not os.path.exists(USERS_DB) or os.path.getsize(USERS_DB) == 0:
        with open(USERS_DB, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["username","password_hash","created"])
            writer.writeheader()

def user_exists(username):
    ensure_users_db()
    with open(USERS_DB, "r") as f:
        for row in csv.DictReader(f):
            if row["username"] == username:
                return True
    return False

def register_user(username, password):
    ensure_users_db()
    if user_exists(username):
        return False
    with open(USERS_DB, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["username","password_hash","created"])
        writer.writerow({
            "username": username,
            "password_hash": hash_password(password),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    create_user_history_file(username)
    return True

def validate_user(username, password):
    ensure_users_db()
    with open(USERS_DB, "r") as f:
        for row in csv.DictReader(f):
            if row["username"] == username and row["password_hash"] == hash_password(password):
                return True
    return False

# ------------------------------------------------------------
# PER-USER HISTORY HANDLING
# ------------------------------------------------------------
def create_user_history_file(username):
    path = os.path.join(HISTORY_DIR, f"{username}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
    return path

def load_user_history(username):
    path = create_user_history_file(username)
    try:
        df = pd.read_csv(path)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[HISTORY_COLUMNS]
        return df
    except Exception:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
        df.to_csv(path, index=False)
        return df

def append_user_history(username, txn):
    path = create_user_history_file(username)
    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.concat([df, pd.DataFrame([txn])], ignore_index=True)
    df = df.reindex(columns=HISTORY_COLUMNS, fill_value=None)
    df.to_csv(path, index=False)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict) and "model" in data:
            return data["model"], data.get("feature_names", None)
        return data, None
    return None, None

model, model_features = load_model()

def predict_prob(txn):
    if model is None:
        return float(np.random.beta(1, 10))  # fallback random probability
    df = pd.DataFrame([txn])
    if model_features:
        for f in model_features:
            if f not in df.columns:
                df[f] = 0
        df = df[model_features]
    df = df.fillna(0)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(df)[0][1])
    return float(model.predict(df)[0])

# ------------------------------------------------------------
# RANDOM LIVE STREAM (NOT user-specific)
# ------------------------------------------------------------
def generate_random_transaction():
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": f"user_{np.random.randint(1,500)}",
        "merchant_id": f"merchant_{np.random.randint(1,300)}",
        "txn_type": np.random.choice(["send","receive"]),
        "upi_app": np.random.choice(["GPay","PhonePe","Paytm","AmazonPay","BHIM"]),
        "amount": round(np.random.uniform(50,30000),2),
        "user_avg_amount": round(np.random.uniform(50,15000),2),
        "user_txn_count": np.random.randint(1,500),
        "user_age_days": np.random.randint(10,2000),
        "user_device_count": np.random.randint(1,6),
        "merchant_risk_score": round(np.random.uniform(0,1),2),
        "hour": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "ip_change": np.random.choice([0,1], p=[0.9,0.1]),
        "device_change": np.random.choice([0,1],p=[0.9,0.1]),
        "txn_velocity": round(np.random.uniform(0,10),2),
        "previous_fraud_count": np.random.choice([0,1,2]),
        "is_new_merchant": np.random.choice([0,1]),
        "geo_distance_km": round(np.random.uniform(1,2000),2)
    }

# ---------------------------
# STREAMLIT SETUP
# ---------------------------
st.set_page_config(page_title="ΛLVΛN UPI Fraud Demo", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False
if "username" not in st.session_state:
    st.session_state.username = None
if "stream_running" not in st.session_state:
    st.session_state.stream_running = False
if "global_stream_history" not in st.session_state:
    st.session_state.global_stream_history = []  # RIGHT SIDE FEED

# ---------------------------
# LOGIN / REGISTER SCREEN
# ---------------------------
if not st.session_state.auth:
    st.title("🔐 UPI Fraud Detection — Login / Register")
    choice = st.radio("Choose:", ["Login", "Register"], horizontal=True)

    if choice == "Login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if validate_user(u, p):
                st.session_state.auth = True
                st.session_state.username = u
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    else:
        u = st.text_input("Create Username")
        p1 = st.text_input("Password", type="password")
        p2 = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if p1 != p2:
                st.error("Passwords don't match.")
            elif user_exists(u):
                st.warning("Username already exists.")
            else:
                register_user(u, p1)
                st.success("Registration complete — now login.")
                st.experimental_rerun()

    st.stop()

# ---------------------------
# MAIN DASHBOARD
# ---------------------------
username = st.session_state.username
st.title(f"ΛLVΛN — UPI Fraud Detection Dashboard")
st.caption(f"Logged in as **{username}**")

if st.button("Logout"):
    st.session_state.auth = False
    st.session_state.username = None
    st.experimental_rerun()

st.markdown("---")

# ---------------------------
# LAYOUT COLUMNS
# ---------------------------
left, right = st.columns([2,1])

# ------------------------------------------------------------
# LEFT SIDE — MANUAL ENTRY + PERSONAL HISTORY
# ------------------------------------------------------------
with left:
    st.subheader("💳 Manual Transaction Entry")

    with st.form("manual_form"):
        txn = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": username,
            "merchant_id": st.text_input("Merchant ID", "merchant_1"),
            "txn_type": st.selectbox("Type", ["send","receive","withdraw","upi_collect"]),
            "upi_app": st.selectbox("UPI App", ["GPay","PhonePe","Paytm","AmazonPay","BHIM"]),
            "amount": st.number_input("Amount", 1.0, 30000.0, 1500.0),
            "user_avg_amount": st.number_input("User Avg Amount", 1.0, 30000.0, 1200.0),
            "user_txn_count": st.number_input("Txn Count", 1, 2000, 25),
            "user_age_days": st.number_input("User Age Days", 1, 5000, 200),
            "user_device_count": st.number_input("Device Count", 1, 10, 2),
            "merchant_risk_score": st.slider("Merchant Risk", 0.0,1.0,0.5),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "ip_change": int(st.checkbox("IP Change")),
            "device_change": int(st.checkbox("Device Change")),
            "txn_velocity": st.number_input("Txn Velocity", 0.0,20.0,1.0),
            "previous_fraud_count": st.number_input("Prev Fraud",0,10,0),
            "is_new_merchant": int(st.checkbox("New Merchant")),
            "geo_distance_km": st.number_input("Geo Distance (km)",0.0,5000.0,10.0)
        }
        submitted = st.form_submit_button("Submit")

    if submitted:
        prob = predict_prob(txn)
        txn["fraud_probability"] = round(prob*100,2)
        txn["fraud_flag"] = "⚠️ FRAUD" if prob >= 0.9 else "✅ SAFE"
        append_user_history(username, txn)
        st.success(f"Saved — Fraud Probability: {txn['fraud_probability']}% ({txn['fraud_flag']})")

    st.subheader("📁 Your Transaction History")
    hist = load_user_history(username)
    if hist.empty:
        st.info("No transactions yet")
    else:
        st.dataframe(hist.sort_values(by="timestamp", ascending=False), height=430)

# ------------------------------------------------------------
# RIGHT SIDE — LIVE RANDOM STREAM (NOT USER SPECIFIC)
# ------------------------------------------------------------
with right:
    st.subheader("📡 Live Transaction Stream (Random Users)")

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Start Stream"):
            st.session_state.stream_running = True
    with colB:
        if st.button("Stop Stream"):
            st.session_state.stream_running = False

    stream_interval = st.number_input("Interval (sec)", 0.1, 5.0, 1.0, 0.1)
    stream_box = st.empty()

    if st.session_state.stream_running:
        # generate a random transaction
        t = generate_random_transaction()
        prob = predict_prob(t)
        t["fraud_probability"] = round(prob*100,2)
        t["fraud_flag"] = "⚠️ FRAUD" if prob >= 0.9 else "✅ SAFE"

        # push into stream buffer
        st.session_state.global_stream_history.insert(0, t)
        st.session_state.global_stream_history = st.session_state.global_stream_history[:20]

        # display feed
        text_log = ""
        for row in st.session_state.global_stream_history:
            text_log += f"{row['timestamp']} | {row['user_id']} → {row['merchant_id']} | ₹{row['amount']} | {row['fraud_flag']} ({row['fraud_probability']}%)\n"

        stream_box.markdown(f"```bash\n{text_log}\n```")

        time.sleep(stream_interval)
        st.experimental_rerun()
    else:
        stream_box.info("Stream is stopped. Press Start Stream.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("Built by ΛLVΛN Tech — Secure, Real-time Fraud Detection Demo")
