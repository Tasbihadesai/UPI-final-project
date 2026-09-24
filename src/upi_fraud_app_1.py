#!/usr/bin/env python3
"""
upi_fraud_app.py — Corrected for LightGBM inference

Fixes:
 - Runs basic_clean() and feature_engineer() before prediction
 - Loads model metadata including feature_names and categorical_features
 - Aligns feature columns exactly as the model expects
 - Casts categorical features to 'category' dtype rather than zeroing them out
 - Keeps admin live stream + stats + users list layout (live top, stats middle, users bottom)
 - Manual user transactions are still added to histories and global stream
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import os
import csv
import time
import base64
from datetime import datetime

# Import preprocessing functions (user-provided file)
from preprocess import basic_clean, feature_engineer

def load_local_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string
# -------------------- BACKGROUND IMAGE (LOCAL) --------------------
img = load_local_background("src/bg.png")  # <-- your actual path

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{img}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}

    /* REMOVE overlay that caused blackout */
    .stApp:before {{
        content: none !important;
    }}

    /* Make text readable on dark images */
    .stMarkdown, .stMetric, .stText, .stHeader {{
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
    /* Hide Streamlit header */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Hide Streamlit toolbar (Deploy, User menu, etc.) */
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- config --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_DB = os.path.join(DATA_DIR, "users.csv")
HISTORY_DIR = os.path.join(DATA_DIR, "user_history")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model.pkl")  # must match your trained model path

STREAM_INTERVAL = 5   # seconds
MAX_STREAM_LEN = 500

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# -------------------- session defaults --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "GLOBAL_STREAM" not in st.session_state:
    st.session_state.GLOBAL_STREAM = []
if "stream_started" not in st.session_state:
    st.session_state.stream_started = False

# -------------------- history columns --------------------
HISTORY_COLUMNS = [
    "timestamp","user_id","merchant_id","txn_type","upi_app","amount",
    "user_avg_amount","user_txn_count","user_age_days","user_device_count",
    "merchant_risk_score","hour","day_of_week","ip_change","device_change",
    "txn_velocity","previous_fraud_count","is_new_merchant","geo_distance_km",
    "fraud_probability","fraud_flag"
]

# -------------------- auth utilities --------------------
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def ensure_users_db():
    if not os.path.exists(USERS_DB):
        with open(USERS_DB, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["username","password_hash","role","created"]).writeheader()

def read_users():
    ensure_users_db()
    try:
        with open(USERS_DB, "r", newline="") as f:
            return [r for r in csv.DictReader(f) if r.get("username")]
    except Exception:
        ensure_users_db()
        return []

def user_exists(username): return any(u["username"] == username for u in read_users())

def register_user(username, password):
    ensure_users_db()
    if user_exists(username):
        return False
    users = read_users()
    role = "admin" if len(users) == 0 else "user"
    with open(USERS_DB, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["username","password_hash","role","created"])
        writer.writerow({
            "username": username,
            "password_hash": hash_password(password),
            "role": role,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    create_user_history_file(username)
    return True

def validate_user(username, password):
    h = hash_password(password)
    for r in read_users():
        if r["username"] == username and r["password_hash"] == h:
            return True
    return False

def get_user_role(username):
    for r in read_users():
        if r["username"] == username:
            return r.get("role", "user")
    return "user"

def registered_users():
    return [u["username"] for u in read_users()]

# -------------------- user history --------------------
def create_user_history_file(username):
    path = os.path.join(HISTORY_DIR, f"{username}.csv")
    if not os.path.exists(path):
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
    return path

def load_user_history(username):
    path = create_user_history_file(username)
    try:
        df = pd.read_csv(path)
        for c in HISTORY_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[HISTORY_COLUMNS]
    except Exception:
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
        return pd.DataFrame(columns=HISTORY_COLUMNS)

def append_user_history(username, txn):
    path = create_user_history_file(username)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.concat([df, pd.DataFrame([txn])], ignore_index=True)
    df = df.reindex(columns=HISTORY_COLUMNS)
    df.to_csv(path, index=False)

# -------------------- model loading & metadata --------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)

            if isinstance(data, dict):

                model = data.get("model")
                feature_names = data.get("feature_names", None)
                categorical_features = data.get("categorical_features", None)
                encoders = data.get("encoders", {})

                return model, feature_names, categorical_features, encoders

            return data, None, None, {}

        except Exception as e:
            st.warning(f"Failed to load model: {e}")
            return None, None, None, {}

    return None, None, None, {}


MODEL, MODEL_FEATURES, MODEL_CAT_FEATURES, ENCODERS = load_model()

def prepare_features_for_model(df_input: pd.DataFrame):
    """
    Run the exact preprocessing and feature engineering used during training,
    then align to MODEL_FEATURES and set categorical dtypes for MODEL_CAT_FEATURES.
    Returns DataFrame ready for model.predict_proba.
    """
    df = df_input.copy()

    # apply same cleaning and feature engineering as training
    try:
        df = basic_clean(df)
        df, _ = feature_engineer(df)

        # Apply same encoding as training
        for col, le in ENCODERS.items():

            if col in df.columns:

                df[col] = df[col].astype(str)

                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0]
                    if x in le.classes_
                    else -1
                )

    except Exception as e:
        st.write("⚠️ preprocess failed:", e)




    except Exception as e:
        # if preprocess fails, at least try to continue with raw df
        st.write("⚠️ preprocess failed:", e)

    # keep only the features the model expects (create missing ones as 0)
    if MODEL_FEATURES:
        for f in MODEL_FEATURES:
            if f not in df.columns:
                df[f] = 0
        X = df.loc[:, MODEL_FEATURES].copy()
    else:
        X = df.copy()

    # Ensure numeric columns are numeric (for safety)
    # For categorical features we'll cast later to 'category' if model provided them
    for col in X.columns:
        if col not in (MODEL_CAT_FEATURES or []):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # If model provided categorical feature names, cast them to 'category' type
    if MODEL_CAT_FEATURES:
        for c in MODEL_CAT_FEATURES:
            if c in X.columns:
                # preserve existing values but cast to category
                X[c] = X[c].astype("category")

    return X

def predict_prob(txn: dict) -> float:
    """
    Returns fraud probability (0..1).
    Uses model metadata to align features and preserve categorical features.
    """
    if MODEL is None:
        return float(np.random.beta(0.8, 8.0))

    df = pd.DataFrame([txn])
    X = prepare_features_for_model(df)

    st.write("STREAMLIT MODEL INPUT")
    st.write(X)

    st.write("MODEL FEATURES")
    st.write(MODEL_FEATURES)

    try:
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(X)
            return float(probs[0][1])
        else:
            return float(MODEL.predict(X)[0])
    except Exception as e:
        st.write("⚠️ Model prediction error:", e)
        return float(np.random.beta(0.8, 8.0))

# -------------------- random txn generator --------------------
def random_stream_txn():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": ts,
        "user_id": f"user_{np.random.randint(1,500)}",
        "merchant_id": f"merchant_{np.random.randint(1,300)}",
        "txn_type": np.random.choice(["send","receive","withdraw","upi_collect"]),
        "upi_app": np.random.choice(["GPay","PhonePe","Paytm","AmazonPay","BHIM","other"]),
        "amount": round(np.random.uniform(50,30000),2),
        "user_avg_amount": round(np.random.uniform(50,15000),2),
        "user_txn_count": np.random.randint(1,500),
        "user_age_days": np.random.randint(10,2000),
        "user_device_count": np.random.randint(1,6),
        "merchant_risk_score": round(np.random.uniform(0,1),2),
        "hour": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "ip_change": int(np.random.choice([0,1], p=[0.9,0.1])),
        "device_change": int(np.random.choice([0,1], p=[0.9,0.1])),
        "txn_velocity": round(np.random.uniform(0,10),2),
        "previous_fraud_count": int(np.random.choice([0,1,2])),
        "is_new_merchant": int(np.random.choice([0,1])),
        "geo_distance_km": round(np.random.uniform(1,5000),2)
    }

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="UPI Fraud Detection", layout="wide")

# Login / register
if not st.session_state.authenticated:
    st.title("🔐 Login / Register")
    action = st.radio("Action", ["Login", "Register"], horizontal=True)
    if action == "Login":
        uname = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if validate_user(uname, pwd):
                st.session_state.authenticated = True
                st.session_state.username = uname
                st.rerun()
            else:
                st.error("Invalid username or password")
    else:
        uname = st.text_input("Choose username")
        p1 = st.text_input("Password", type="password")
        p2 = st.text_input("Confirm password", type="password")
        if st.button("Register"):
            if p1 != p2:
                st.error("Passwords do not match")
            elif user_exists(uname):
                st.error("User exists")
            else:
                register_user(uname, p1)
                st.success("Registered — first user becomes admin")
                st.rerun()
    st.stop()

# After login
username = st.session_state.username
role = get_user_role(username)

st.title("UPI Fraud Detection")
st.caption(f"Signed in as **{username}** (role: **{role}**)")
if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.stream_started = False
    st.rerun()

st.markdown("---")

# User manual txn flow
if role == "user":
    st.subheader("💳 Manual Transaction")
    recipients = registered_users()
    if username in recipients:
        recipients.remove(username)
    with st.form("manual_txn"):
        recipient = st.selectbox("Send To", recipients if recipients else ["merchant_1"])
        txn_type = st.selectbox("Txn Type", ["send","receive","withdraw","upi_collect"])
        upi_app = st.selectbox("UPI App", ["GPay","PhonePe","Paytm","AmazonPay","BHIM","other"])
        amount = st.number_input("Amount (₹)", 1.0, 30000.0, 1500.0, step=0.01)
        user_avg_amount = st.number_input("User avg amount", 0.0, 30000.0, 1200.0, step=0.01)
        user_txn_count = int(st.number_input("User txn count", 1, 10000, 25))
        user_age_days = int(st.number_input("User age (days)", 1, 5000, 200))
        user_device_count = int(st.number_input("User device count", 1, 10, 2))
        merchant_risk_score = float(st.slider("Merchant risk", 0.0, 1.0, 0.35))
        ip_change = int(st.checkbox("IP change"))
        device_change = int(st.checkbox("Device change"))
        txn_velocity = float(st.number_input("Txn velocity", 0.0, 100.0, 1.5))
        previous_fraud_count = int(st.number_input("Previous fraud count", 0, 20, 0))
        is_new_merchant = int(st.checkbox("New merchant"))
        geo_distance_km = float(st.number_input("Distance (km)", 0.0, 10000.0, 10.0))
        submitted = st.form_submit_button("Send")
    if submitted:
        txn = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": username,
            "merchant_id": recipient,
            "txn_type": txn_type,
            "upi_app": upi_app,
            "amount": round(float(amount),2),
            "user_avg_amount": round(float(user_avg_amount),2),
            "user_txn_count": int(user_txn_count),
            "user_age_days": int(user_age_days),
            "user_device_count": int(user_device_count),
            "merchant_risk_score": float(merchant_risk_score),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "ip_change": int(ip_change),
            "device_change": int(device_change),
            "txn_velocity": float(txn_velocity),
            "previous_fraud_count": int(previous_fraud_count),
            "is_new_merchant": int(is_new_merchant),
            "geo_distance_km": float(geo_distance_km),
        }
        prob = predict_prob(txn)
        txn["fraud_probability"] = round(prob*100.0,2)
        txn["fraud_flag"] = "⚠️ FRAUD" if prob >= 0.98 else "✅ SAFE"
        append_user_history(username, txn)
        if recipient in registered_users():
            append_user_history(recipient, txn)
        st.session_state.GLOBAL_STREAM.insert(0, txn)
        if len(st.session_state.GLOBAL_STREAM) > MAX_STREAM_LEN:
            st.session_state.GLOBAL_STREAM = st.session_state.GLOBAL_STREAM[:MAX_STREAM_LEN]
        st.success(f"Transaction recorded — Fraud {txn['fraud_probability']}%")
    st.markdown("---")
    st.subheader("📁 Your History")
    uh = load_user_history(username)
    if uh.empty:
        st.info("No transactions yet.")
    else:
        display = uh.copy()
        try:
            display["amount"] = display["amount"].apply(lambda x: f"₹{float(x):,.2f}" if pd.notna(x) else x)
            display["fraud_probability"] = display["fraud_probability"].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else x)
        except Exception:
            pass
        st.dataframe(display.sort_values("timestamp", ascending=False), use_container_width=True, height=450)
    st.stop()

# Admin view: live top, stats middle, users bottom
st.subheader("📡 Live Transactions")
live_placeholder = st.empty()

st.subheader("📊 Global Statistics")
stats_placeholder = st.empty()

st.markdown("---")

st.subheader("👥 All Users & Their Transaction Histories")
users = registered_users()
users_df = pd.DataFrame([{"username": u, "role": get_user_role(u)} for u in users])
st.table(users_df)
sel = st.selectbox("Select a user to view history", ["--choose--"] + users)
if sel != "--choose--":
    h = load_user_history(sel)
    if h.empty:
        st.info("No transactions for that user.")
    else:
        hv = h.copy()
        try:
            hv["amount"] = hv["amount"].apply(lambda x: f"₹{float(x):,.2f}")
            hv["fraud_probability"] = hv["fraud_probability"].apply(lambda x: f"{float(x):.2f}%")
        except Exception:
            pass
        st.dataframe(hv.sort_values("timestamp", ascending=False), height=400)

st.markdown("---")

# start stream loop once
if not st.session_state.stream_started:
    st.session_state.stream_started = True
    st.rerun()

# Infinite loop: updates top (live) and middle (stats)
while True:
    try:
        # new random tx
        tx = random_stream_txn()
        prob = predict_prob(tx)
        tx["fraud_probability"] = round(prob * 100.0, 2)
        tx["fraud_flag"] = "⚠️ FRAUD" if prob >= 0.98 else "✅ SAFE"

        st.session_state.GLOBAL_STREAM.insert(0, tx)
        if len(st.session_state.GLOBAL_STREAM) > MAX_STREAM_LEN:
            st.session_state.GLOBAL_STREAM = st.session_state.GLOBAL_STREAM[:MAX_STREAM_LEN]

        # LIVE (top)
        live_lines = []
        for r in st.session_state.GLOBAL_STREAM[:60]:
            live_lines.append(f"{r.get('timestamp','')} | {r.get('user_id','')} → {r.get('merchant_id','')} | ₹{r.get('amount','')} | {r.get('fraud_flag','')} ")
        with live_placeholder:
            st.code("\n".join(live_lines), language="bash")

        # STATS (middle)
        hist_tx = 0
        hist_flag = 0
        hist_probs = []
        for u in users:
            df = load_user_history(u)
            if df is None or df.empty:
                continue
            hist_tx += len(df)
            try:
                hist_flag += int((df["fraud_flag"] == "⚠️ FRAUD").sum())
            except Exception:
                pass
            try:
                hist_probs.extend([float(x) for x in df["fraud_probability"].dropna().tolist()])
            except Exception:
                pass

        stream_tx = len(st.session_state.GLOBAL_STREAM)
        stream_flag = sum(1 for x in st.session_state.GLOBAL_STREAM if x.get("fraud_flag") == "⚠️ FRAUD")
        stream_probs = [x.get("fraud_probability", 0) for x in st.session_state.GLOBAL_STREAM]

        total_tx = hist_tx + stream_tx
        total_flag = hist_flag + stream_flag
        all_probs = hist_probs + stream_probs
        avg_prob = float(np.mean(all_probs)) if len(all_probs) > 0 else 0.0

        with stats_placeholder:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", total_tx)
            c2.metric("Total Flagged", total_flag)
            c3.metric("Avg Fraud Probability", f"{avg_prob:.2f}%")

    except Exception as e:
        with live_placeholder:
            st.error(f"Stream error: {e}")

    time.sleep(STREAM_INTERVAL)
