#!/usr/bin/env python3
"""
upi_fraud_app.py — Final stable single-file Streamlit app for UPI Fraud demo.

Page flow (role == "user"):
    Login -> Dashboard -> Transaction -> Result -> History

Page flow (role == "admin"):
    Login -> Admin Overview -> All Users -> Model Performance
          -> Model Comparison -> Live Stream
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import os
import csv
import time
import uuid
from datetime import datetime
from explain import explain_transaction
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import requests
import networkx as nx

# ------------------------------
# Global chart styling (applies to every matplotlib chart in this app)
# ------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0A1B33",
    "axes.facecolor": "#0A1B33",
    "axes.edgecolor": "#8FA8C7",
    "axes.labelcolor": "#FFFFFF",
    "text.color": "#FFFFFF",
    "xtick.color": "#C8D6EA",
    "ytick.color": "#C8D6EA",
    "grid.color": "#1E3A5F",
    "grid.alpha": 0.3,
    "axes.grid": True,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "legend.facecolor": "#0A1B33",
    "legend.edgecolor": "#8FA8C7",
    "legend.labelcolor": "#FFFFFF",
    "savefig.facecolor": "#0A1B33",
})

# ------------------------------
# Config / Paths / Constants
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_DB = os.path.join(DATA_DIR, "users.csv")
HISTORY_DIR = os.path.join(DATA_DIR, "user_history")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model.pkl")

STREAM_INTERVAL = 5
MAX_STREAM_LEN = 500

# Logo / hero image — resolved relative to this script's own folder so it
# works no matter what directory Streamlit is launched from.
BG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg.png")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ------------------------------
# Session state defaults
# ------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "GLOBAL_STREAM" not in st.session_state:
    st.session_state.GLOBAL_STREAM = []
if "stream_started" not in st.session_state:
    st.session_state.stream_started = False
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "admin_page" not in st.session_state:
    st.session_state.admin_page = "overview"
if "entry_page" not in st.session_state:
    st.session_state.entry_page = "landing"
if "login_mode" not in st.session_state:
    st.session_state.login_mode = "user"
if "auth_action" not in st.session_state:
    st.session_state.auth_action = "login"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_chat_widget" not in st.session_state:
    st.session_state.show_chat_widget = False

# ------------------------------
# History csv columns
# ------------------------------
HISTORY_COLUMNS = [
    "timestamp","user_id","merchant_id","txn_type","upi_app","amount",
    "user_avg_amount","user_txn_count","user_age_days","user_device_count",
    "merchant_risk_score","hour","day_of_week","ip_change","device_change",
    "txn_velocity","previous_fraud_count","is_new_merchant","geo_distance_km",
    "fraud_probability","fraud_flag",
    "risk_level","transaction_status","payment_status","receiver_credited",
    "api_message"
]

# ------------------------------
# Utilities: auth and users
# ------------------------------
def hash_password(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

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

def user_exists(username: str) -> bool:
    return any(u["username"] == username for u in read_users())

def register_user(username: str, password: str) -> bool:
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

def validate_user(username: str, password: str) -> bool:
    h = hash_password(password)
    for r in read_users():
        if r["username"].strip() == username.strip() and r["password_hash"].strip() == h.strip():
            return True
    return False

def reset_user_password(username: str, new_password: str) -> bool:
    """
    Resets a user's password in-place, preserving every other account and
    the existing CSV schema/column order. Returns False if the username
    doesn't exist.
    """
    users = read_users()
    found = False
    for u in users:
        if u["username"] == username:
            u["password_hash"] = hash_password(new_password)
            found = True
            break

    if not found:
        return False

    fieldnames = ["username", "password_hash", "role", "created"]
    with open(USERS_DB, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for u in users:
            writer.writerow({k: u.get(k, "") for k in fieldnames})
    return True

def get_user_role(username: str) -> str:
    for r in read_users():
        if r["username"] == username:
            return r.get("role", "user")
    return "user"

def registered_users() -> list:
    return [u["username"] for u in read_users()]

# ------------------------------
# User history helpers
# ------------------------------
def create_user_history_file(username: str) -> str:
    path = os.path.join(HISTORY_DIR, f"{username}.csv")
    if not os.path.exists(path):
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)
    return path

def load_user_history(username: str) -> pd.DataFrame:
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

def append_user_history(username: str, txn: dict):
    path = create_user_history_file(username)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.concat([df, pd.DataFrame([txn])], ignore_index=True)
    df = df.reindex(columns=HISTORY_COLUMNS)
    df.to_csv(path, index=False)

# ------------------------------
# Fraud alert audit log
# ------------------------------
FRAUD_ALERTS_PATH = os.path.join(DATA_DIR, "fraud_alerts.csv")
ALERT_COLUMNS = ["alert_id"] + HISTORY_COLUMNS + ["reviewed"]

def ensure_fraud_alerts_db():
    if not os.path.exists(FRAUD_ALERTS_PATH):
        pd.DataFrame(columns=ALERT_COLUMNS).to_csv(FRAUD_ALERTS_PATH, index=False)

def append_fraud_alert(txn: dict):
    """Logs a transaction to the fraud alert audit trail — only called when fraud_flag is FRAUD."""
    if txn.get("fraud_flag") != "⚠️ FRAUD":
        return
    ensure_fraud_alerts_db()
    alert_row = dict(txn)
    alert_row["alert_id"] = uuid.uuid4().hex[:10]
    alert_row["reviewed"] = "No"
    try:
        df = pd.read_csv(FRAUD_ALERTS_PATH)
    except Exception:
        df = pd.DataFrame(columns=ALERT_COLUMNS)
    df = pd.concat([df, pd.DataFrame([alert_row])], ignore_index=True)
    df = df.reindex(columns=ALERT_COLUMNS)
    df.to_csv(FRAUD_ALERTS_PATH, index=False)

def load_fraud_alerts() -> pd.DataFrame:
    ensure_fraud_alerts_db()
    try:
        df = pd.read_csv(FRAUD_ALERTS_PATH)
        for c in ALERT_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[ALERT_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=ALERT_COLUMNS)

def mark_alert_reviewed(alert_id: str):
    df = load_fraud_alerts()
    if "alert_id" in df.columns:
        df.loc[df["alert_id"] == alert_id, "reviewed"] = "Yes"
        df.to_csv(FRAUD_ALERTS_PATH, index=False)

# ------------------------------
# Model loading & safe predict
# ------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            if isinstance(data, dict):
                return (
                    data.get("model"),
                    data.get("feature_names", None),
                    data.get("encoders", {})
                )
            return data, None, {}
        except Exception as e:
            print("Model loading error:", e)
            return None, None, {}
    return None, None, {}

MODEL, MODEL_FEATURES, ENCODERS = load_model()

FEATURE_LABELS = {
    "amount": "Transaction Amount",
    "user_avg_amount": "User's Average Amount",
    "user_txn_count": "User's Transaction Count",
    "user_age_days": "Account Age (days)",
    "user_device_count": "Number of Devices Used",
    "merchant_risk_score": "Merchant Risk Score",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "ip_change": "IP Address Changed",
    "device_change": "Device Changed",
    "txn_velocity": "Transaction Velocity",
    "previous_fraud_count": "Previous Fraud Count",
    "is_new_merchant": "New Merchant",
    "geo_distance_km": "Geographic Distance (km)",
    "user_total_txn_count": "Total Transactions by User",
    "amount_dev_from_avg": "Deviation from Average Amount",
    "log_amount": "Log-Scaled Amount",
    "is_odd_hour": "Unusual Hour",
    "device_ip_combo": "Device + IP Combination",
    "user_behavior_risk": "User Behavior Risk Score",
}

def pretty_feature_name(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())

def _prepare_features(txn: dict) -> pd.DataFrame:
    df = pd.DataFrame([txn])
    from preprocess import basic_clean, feature_engineer
    df = basic_clean(df)
    df, _ = feature_engineer(df)

    for col, le in ENCODERS.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    if MODEL_FEATURES:
        for f in MODEL_FEATURES:
            if f not in df.columns:
                df[f] = 0
        df = df[MODEL_FEATURES]

    return df.fillna(0)

def predict_prob(txn: dict) -> float:
    if MODEL is None:
        return float(np.random.beta(0.8, 8.0))
    try:
        df = _prepare_features(txn)
        if hasattr(MODEL, "predict_proba"):
            return float(MODEL.predict_proba(df)[0][1])
        return float(MODEL.predict(df)[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0

def predict_transaction_api(txn: dict) -> dict:
    """
    Sends the transaction to the Flask API and returns the
    complete fraud decision, including transaction/payment status.
    """
    try:
        response = requests.post(
            "http://127.0.0.1:5000/transaction",
            json=txn,
            timeout=10
        )

        # Treat API errors as failures instead of silently
        # converting them into a SAFE prediction.
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}")
        return {
            "success": False,
            "error": "Unable to connect to fraud detection API.",
            "transaction_status": "NOT_APPROVED",
            "payment_status": "NOT_EXECUTED",
            "receiver_credited": False
        }

    except Exception as e:
        st.error(f"Unexpected API Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "transaction_status": "NOT_APPROVED",
            "payment_status": "NOT_EXECUTED",
            "receiver_credited": False
        }


@st.cache_resource
def load_lime_explainer():
    """
    Builds a LIME tabular explainer once, using a saved sample of the
    real training data as background so LIME understands normal value
    ranges for each feature (unlike SHAP's TreeExplainer, LIME needs this).
    """
    import lime
    import lime.lime_tabular

    bg_path = os.path.join(BASE_DIR, "models", "lime_background.csv")
    if not os.path.exists(bg_path) or not MODEL_FEATURES:
        return None

    try:
        bg_df = pd.read_csv(bg_path)
        bg_df = bg_df.reindex(columns=MODEL_FEATURES, fill_value=0).fillna(0)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_df.values,
            feature_names=MODEL_FEATURES,
            class_names=["Safe", "Fraud"],
            mode="classification",
            discretize_continuous=True,
        )
        return explainer
    except Exception as e:
        st.warning(f"LIME explainer unavailable: {e}")
        return None

def get_lime_explanation(txn: dict, top_n: int = 5):
    """
    Returns a list of (feature_description, weight) tuples from LIME for
    this single transaction, run against the same LightGBM model used
    for the prediction and for SHAP.
    """
    if MODEL is None:
        return None

    explainer = load_lime_explainer()
    if explainer is None:
        return None

    try:
        df = _prepare_features(txn)

        def predict_fn(X):
            X_df = pd.DataFrame(X, columns=df.columns)
            if hasattr(MODEL, "predict_proba"):
                return MODEL.predict_proba(X_df)
            preds = MODEL.predict(X_df)
            return np.column_stack([1 - preds, preds])

        exp = explainer.explain_instance(
            df.iloc[0].values,
            predict_fn,
            num_features=top_n,
        )
        return exp.as_list()
    except Exception as e:
        st.warning(f"LIME explanation unavailable: {e}")
        return None

def get_recommendation(txn: dict, analysis: dict, shap_result=None, lime_result=None) -> dict:
    """
    Turns the raw fraud probability + risk signals into a single,
    actionable recommendation for the admin/reviewer: whether to
    block the user, hold the transaction for manual review, or
    let it proceed — plus the reasoning behind that call.
    """
    prob = analysis["probability"]
    prev_fraud = int(txn.get("previous_fraud_count", 0) or 0)
    ip_change = int(txn.get("ip_change", 0) or 0)
    device_change = int(txn.get("device_change", 0) or 0)
    is_blocked = txn.get("transaction_status") == "BLOCKED"

    # Check whether SHAP and LIME agree on the single strongest signal —
    # used as a confidence booster/reducer on the recommendation.
    top_shap_feature = None
    if shap_result is not None and len(shap_result) > 0:
        top_shap_feature = shap_result.index[0]

    top_lime_feature = None
    if lime_result:
        top_lime_feature = lime_result[0][0]

    methods_agree = bool(
        top_shap_feature
        and top_lime_feature
        and pretty_feature_name(top_shap_feature).lower() in top_lime_feature.lower()
    )

    # IMPORTANT: the recommendation must never contradict the transaction
    # decision the Flask API already made. If the payment was blocked and
    # not executed, the recommendation can only ever be some flavor of
    # "block/escalate" — never "hold for review" or "proceed", since those
    # would misrepresent what actually happened to the money.
    if is_blocked:
        action = "🚫 Block User"
        action_color = "#e74c3c"
        confidence = "High" if (methods_agree or prob >= 90) else "Medium-High"
        justification = (
            f"This transaction was already blocked by the fraud API (fraud probability {prob:.2f}%)"
            + (f", and this user has {prev_fraud} prior fraud incident(s)" if prev_fraud > 0 else "")
            + ". Recommend blocking the user and escalating for manual investigation before any further transactions are allowed."
        )
    elif prob >= 30 or prev_fraud > 0:
        action = "✅ Proceed with Monitoring"
        action_color = "#4FC3F7"
        confidence = "Medium"
        justification = (
            f"Transaction was approved (fraud probability {prob:.2f}%, below the block threshold)"
            + (", combined with a recent IP or device change" if (ip_change or device_change) else "")
            + ", but flag this account for closer monitoring on subsequent transactions."
        )
    else:
        action = "✅ Proceed with Transaction"
        action_color = "#2ecc71"
        confidence = "High"
        justification = (
            f"Fraud probability is low ({prob:.2f}%) with no material risk signals detected. "
            "Safe to proceed with the transaction as normal."
        )

    return {
        "action": action,
        "color": action_color,
        "confidence": confidence,
        "justification": justification,
        "methods_agree": methods_agree,
    }

def get_shap_explanation(txn: dict, top_n: int = 5):
    if MODEL is None:
        return None
    try:
        df = _prepare_features(txn)
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(df)
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        shap_series = pd.Series(values, index=df.columns).sort_values(key=abs, ascending=False)
        return shap_series.head(top_n)
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
        return None

# ------------------------------
# QR scan helpers (used by the Scan-to-Pay feature)
# ------------------------------
def parse_upi_qr(data: str) -> dict:
    """Parses a decoded QR string. Handles real UPI URIs (upi://pay?pa=...&pn=...)
    and falls back to treating the raw string as the identifier for any other QR content."""
    from urllib.parse import urlparse, parse_qs
    if data and data.startswith("upi://"):
        parsed = urlparse(data)
        params = parse_qs(parsed.query)
        return {
            "raw": data,
            "vpa": params.get("pa", [None])[0],
            "name": params.get("pn", [None])[0],
            "amount": params.get("am", [None])[0],
        }
    return {"raw": data, "vpa": data, "name": None, "amount": None}

def decode_qr_from_image(uploaded_file) -> str:
    """Decodes a QR code from an uploaded image file (Streamlit UploadedFile). Returns the
    decoded string, or None if no QR code was found or the libraries aren't available."""
    try:
        import cv2
        import numpy as _np
        from pyzbar.pyzbar import decode as qr_decode
    except ImportError:
        st.error("QR scanning requires 'opencv-python' and 'pyzbar'. Install with: pip install opencv-python pyzbar")
        return None

    try:
        file_bytes = _np.asarray(bytearray(uploaded_file.read()), dtype=_np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None

        decoded_objects = qr_decode(img)
        if not decoded_objects:
            return None
        return decoded_objects[0].data.decode("utf-8")
    except Exception as e:
        st.error(f"Could not read QR code from image: {e}")
        return None

def scan_qr_live() -> str:
    """Opens a native webcam window (same approach as test_camera.py) and returns the
    first decoded QR code string, or None if cancelled (ESC) or unavailable."""
    try:
        import cv2
        from pyzbar.pyzbar import decode as qr_decode
    except ImportError:
        st.error("Live scanning requires 'opencv-python' and 'pyzbar'. Install with: pip install opencv-python pyzbar")
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Check that no other app is using it.")
        return None

    result = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            decoded_objects = qr_decode(frame)
            if decoded_objects:
                result = decoded_objects[0].data.decode("utf-8")
                break

            cv2.imshow("Scan QR Code - press ESC to cancel", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
    except Exception as e:
        st.error(f"Camera scan error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return result

def check_merchant_trust(merchant_identifier: str) -> dict:
    """Looks up a scanned merchant identifier across all known transaction history and
    fraud alerts to determine whether this merchant is new, safe, or risky."""
    total_transactions = 0
    fraud_count = 0
    probs = []

    for u in registered_users():
        hist = load_user_history(u)
        if hist is None or hist.empty:
            continue
        matches = hist[hist["merchant_id"] == merchant_identifier]
        if matches.empty:
            continue
        total_transactions += len(matches)
        try:
            fraud_count += int((matches["fraud_flag"] == "⚠️ FRAUD").sum())
        except Exception:
            pass
        try:
            probs.extend([float(x) for x in matches["fraud_probability"].dropna().tolist()])
        except Exception:
            pass

    avg_prob = float(np.mean(probs)) if probs else 0.0

    if total_transactions == 0:
        verdict = "new"
    elif fraud_count > 0:
        verdict = "risky"
    else:
        verdict = "safe"

    return {
        "identifier": merchant_identifier,
        "total_transactions": total_transactions,
        "fraud_count": fraud_count,
        "avg_fraud_probability": avg_prob,
        "verdict": verdict,
    }

# ------------------------------
# Random live tx generator
# ------------------------------
def random_stream_txn() -> dict:
    """
    Generates a simulated transaction for the admin Live Stream demo —
    now grounded in the SAME risk-computation engine real transactions use
    (compute_user_risk_profile), picking an actual registered account as
    the sender so the behavioral/merchant signals reflect real stored
    history instead of being arbitrary random numbers disconnected from
    the rest of the system.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    users_pool = registered_users()

    if not users_pool:
        # Fallback for a brand-new install with no registered accounts yet.
        return {
            "timestamp": timestamp,
            "user_id": f"user_{np.random.randint(1, 500)}",
            "merchant_id": f"merchant_{np.random.randint(1, 300)}",
            "txn_type": np.random.choice(["send", "receive", "withdraw", "upi_collect"]),
            "upi_app": np.random.choice(["GPay", "PhonePe", "Paytm", "AmazonPay", "BHIM"]),
            "amount": round(float(np.random.uniform(50, 30000)), 2),
            "user_avg_amount": round(float(np.random.uniform(50, 15000)), 2),
            "user_txn_count": int(np.random.randint(1, 500)),
            "user_age_days": int(np.random.randint(10, 2000)),
            "user_device_count": int(np.random.randint(1, 6)),
            "merchant_risk_score": round(float(np.random.uniform(0, 1)), 2),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "ip_change": int(np.random.choice([0, 1], p=[0.9, 0.1])),
            "device_change": int(np.random.choice([0, 1], p=[0.9, 0.1])),
            "txn_velocity": round(float(np.random.uniform(0, 10)), 2),
            "previous_fraud_count": int(np.random.choice([0, 1, 2])),
            "is_new_merchant": int(np.random.choice([0, 1])),
            "geo_distance_km": round(float(np.random.uniform(1, 5000)), 2),
        }

    sender = str(np.random.choice(users_pool))
    other_users = [u for u in users_pool if u != sender]

    # ~20% of simulated transactions go to a genuinely unverified/external
    # recipient — same risk framing as the real "Send to someone not in
    # ALVAN" option, so the live feed shows that risk pattern too.
    if other_users and np.random.random() > 0.20:
        recipient = str(np.random.choice(other_users))
    else:
        recipient = f"unknown_{np.random.randint(1000, 9999)}"

    amount = round(float(np.random.uniform(50, 30000)), 2)
    risk_profile = compute_user_risk_profile(sender, recipient, amount)

    return {
        "timestamp": timestamp,
        "user_id": sender,
        "merchant_id": recipient,
        "txn_type": np.random.choice(["send", "receive", "withdraw", "upi_collect"]),
        "upi_app": np.random.choice(["GPay", "PhonePe", "Paytm", "AmazonPay", "BHIM"]),
        "amount": amount,
        "hour": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        **risk_profile,
    }

# ==========================================================
# STREAMLIT LAYOUT & PAGES
# ==========================================================
st.set_page_config(page_title="ΛLVΛN — UPI Fraud Live", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0A4DA3 0%, #062852 45%, #01152e 100%);
}
h1, h2, h3, p, label, .stMarkdown, .stRadio label, .stTextInput label {
    color: #FFFFFF !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 20px;
    padding: 10px 10px 25px 10px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.35);
}
div[data-testid="stVerticalBlockBorderWrapper"] label,
div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3,
div[data-testid="stVerticalBlockBorderWrapper"] h4,
div[data-testid="stVerticalBlockBorderWrapper"] p {
    color: #FFFFFF !important;
}
.stTextInput>div>div>input {
    background: #0F2A4A;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.4);
    padding: 10px;
    color: #FFFFFF !important;
    caret-color: #FFFFFF;
}
.stTextInput>div>div>input::placeholder {
    color: #9FB3CC;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1B33 0%, #01152e 100%);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #0A4DA3, #1668C9);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 0px;
    font-weight: 600;
    font-size: 16px;
    transition: 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #063a80, #0A4DA3);
    transform: scale(1.01);
}
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: linear-gradient(90deg, #0A4DA3, #1668C9);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 0px;
    font-weight: 600;
    font-size: 16px;
    transition: 0.2s;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #063a80, #0A4DA3);
    transform: scale(1.01);
}
div[data-testid="stFormSubmitButton"] button p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# PAGE 1a: LANDING PAGE  (marketing-style front page)
# ----------------------------------------------------------
if not st.session_state.authenticated and st.session_state.entry_page == "landing":

    # ---- Top Nav Bar ----
    nav_l, nav_m, nav_r1, nav_r2 = st.columns([2.2, 2, 0.9, 0.9])
    with nav_l:
        st.markdown("<h2 style='color:#FFFFFF; margin-top:4px;'>🛡️ ALVAN</h2>", unsafe_allow_html=True)
    with nav_r1:
        if st.button("Log In", use_container_width=True):
            st.session_state.entry_page = "login"
            st.rerun()
    with nav_r2:
        if st.button("Sign Up", use_container_width=True):
            st.session_state.entry_page = "login"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Hero Section ----
    hero_left, hero_right = st.columns([1, 1.35])

    with hero_left:
        st.markdown(
            "<h1 style='font-size:46px; color:#FFFFFF; line-height:1.25;'>"
            "Secure UPI Payments,<br>Detected Instantly.</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:18px; color:#DDE6F5; max-width:520px;'>"
            "ALVAN uses Machine Learning and Explainable AI to monitor every "
            "UPI transaction in real time — catching fraud before it costs you.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Get Started", key="hero_cta"):
            st.session_state.entry_page = "login"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.markdown(
                "<h2 style='color:#4FC3F7; margin-bottom:0;'>99.5%</h2>"
                "<p style='color:#DDE6F5; margin-top:0;'>Model Accuracy</p>",
                unsafe_allow_html=True,
            )
        with stat2:
            st.markdown(
                "<h2 style='color:#4FC3F7; margin-bottom:0;'>30,000+</h2>"
                "<p style='color:#DDE6F5; margin-top:0;'>Transactions Analyzed</p>",
                unsafe_allow_html=True,
            )
        with stat3:
            st.markdown(
                "<h2 style='color:#4FC3F7; margin-bottom:0;'>Real-Time</h2>"
                "<p style='color:#DDE6F5; margin-top:0;'>Fraud Detection</p>",
                unsafe_allow_html=True,
            )

    with hero_right:
        if os.path.exists(BG_PATH):
            st.image(BG_PATH, use_container_width=True)

    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ---- Feature Highlights ----
    st.markdown(
        "<h3 style='text-align:center; color:#FFFFFF; margin-top:10px; margin-bottom:25px;'>Why ALVAN?</h3>",
        unsafe_allow_html=True,
    )

    def feature_badge(emoji, color):
        return f"""
        <div style='
            width:60px; height:60px; border-radius:50%;
            background:{color};
            display:flex; align-items:center; justify-content:center;
            margin: 0 auto 12px auto;
        '>
            <span style='font-size:28px;'>{emoji}</span>
        </div>
        """

    f1, f2, f3 = st.columns(3)
    with f1:
        with st.container(border=True):
            st.markdown(feature_badge("🤖", "#1668C9"), unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Machine Learning</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center;'>Trained on real transaction patterns with 99%+ ROC AUC on held-out data.</p>",
                unsafe_allow_html=True,
            )
    with f2:
        with st.container(border=True):
            st.markdown(feature_badge("🧠", "#8E5CF7"), unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Explainable AI</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center;'>Every decision comes with a clear, human-readable explanation via SHAP.</p>",
                unsafe_allow_html=True,
            )
    with f3:
        with st.container(border=True):
            st.markdown(feature_badge("📡", "#E67E22"), unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Live Monitoring</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center;'>Transactions are analyzed and flagged the moment they happen.</p>",
                unsafe_allow_html=True,
            )

    # ---- How It Works ----
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<h3 style='text-align:center; color:#FFFFFF; margin-bottom:25px;'>How It Works</h3>",
        unsafe_allow_html=True,
    )

    step_badge_color = "#1668C9"

    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.markdown(feature_badge("1️⃣", step_badge_color), unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Register</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center;'>Create your account in seconds — no paperwork, no waiting.</p>",
                unsafe_allow_html=True,
            )
    with s2:
        with st.container(border=True):
            st.markdown(feature_badge("2️⃣", step_badge_color), unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Send a Transaction</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center;'>Submit a UPI transaction just like you would on any payment app.</p>",
                unsafe_allow_html=True,
            )
    with s3:
        with st.container(border=True):
            st.markdown(feature_badge("3️⃣", step_badge_color), unsafe_allow_html=True)
            st.markdown("<h4 style='text-align:center;'>Get an Instant Verdict</h4>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align:center;'>Our AI analyzes it in real time and explains exactly why it's Safe or Fraud.</p>",
                unsafe_allow_html=True,
            )

    st.stop()

# ----------------------------------------------------------
# PAGE 1b: LOGIN / REGISTER  (separate page from the landing page)
# ----------------------------------------------------------
if not st.session_state.authenticated and st.session_state.entry_page == "login":

    if st.button("⬅ Back to Home"):
        st.session_state.entry_page = "landing"
        st.rerun()

    st.markdown("<h1 style='text-align:center; color:#FFFFFF;'>ALVAN</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center; color:#DDE6F5;'>AI Powered UPI Fraud Detection System</h3>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    left, mid, right = st.columns([1, 1.6, 1])

    with mid:
        with st.container(border=True):

            act1, act2 = st.columns(2)
            with act1:
                if st.button(
                    "🔑 Login", use_container_width=True,
                    type="primary" if st.session_state.auth_action == "login" else "secondary"
                ):
                    st.session_state.auth_action = "login"
                    st.rerun()
            with act2:
                if st.button(
                    "📝 Register", use_container_width=True,
                    type="primary" if st.session_state.auth_action == "register" else "secondary"
                ):
                    st.session_state.auth_action = "register"
                    st.rerun()

            action = st.session_state.auth_action

            if action == "login":

                avatar_col, form_col = st.columns([1, 1.6])

                with avatar_col:
                    st.markdown(
                        """
                        <div style='
                            width:120px; height:120px; border-radius:50%;
                            background: linear-gradient(135deg, #7B61FF, #4C3BCF);
                            display:flex; align-items:center; justify-content:center;
                            margin: 55px auto 0 auto;
                        '>
                            <span style='font-size:58px;'>🧑</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with form_col:

                    tab1, tab2 = st.columns(2)
                    with tab1:
                        if st.button(
                            "👤 User Login", use_container_width=True,
                            type="primary" if st.session_state.login_mode == "user" else "secondary"
                        ):
                            st.session_state.login_mode = "user"
                            st.rerun()
                    with tab2:
                        if st.button(
                            "🛡️ Admin Login", use_container_width=True,
                            type="primary" if st.session_state.login_mode == "admin" else "secondary"
                        ):
                            st.session_state.login_mode = "admin"
                            st.rerun()

                    if st.session_state.login_mode == "user":
                        st.markdown("#### User Verification")
                    else:
                        st.markdown("#### Admin Verification")

                    uname = st.text_input("Name")
                    pwd = st.text_input("Password", type="password", key="login_password")

                    if st.button("Login", use_container_width=True):
                        if validate_user(uname, pwd):
                            actual_role = get_user_role(uname)
                            if st.session_state.login_mode == "admin" and actual_role != "admin":
                                st.error("This account is not registered as an Admin. Please use User Login instead.")
                            elif st.session_state.login_mode == "user" and actual_role != "user":
                                st.error("This is an Admin account. Please use Admin Login instead.")
                            else:
                                st.session_state.authenticated = True
                                st.session_state.username = uname
                                st.session_state.page = "dashboard"
                                st.session_state.admin_page = "overview"
                                st.success("Login Successful")
                                st.rerun()
                        else:
                            st.error("Invalid username or password")

                    if st.button("Forgot password?", key="forgot_password_link"):
                        st.session_state.auth_action = "forgot_password"
                        st.rerun()

            elif action == "register":
                st.markdown("#### Create a new account ✨")
                uname = st.text_input("👤 Choose username")
                p1 = st.text_input("🔒 Password", type="password", key="register_password")
                p2 = st.text_input("🔒 Confirm password", type="password", key="confirm_password")

                if st.button("Register"):
                    if p1 != p2:
                        st.error("Passwords do not match")
                    elif user_exists(uname):
                        st.error("User already exists")
                    else:
                        register_user(uname, p1)
                        st.success("Registration Successful")
                        st.rerun()

            else:  # action == "forgot_password"
                st.markdown("#### 🔓 Reset Your Password")
                st.caption(
                    "Enter your username and choose a new password. "
                    "(In a production system this step would be verified by email or OTP first.)"
                )
                r_uname = st.text_input("Username", key="reset_username")
                r_p1 = st.text_input("New password", type="password", key="reset_password_1")
                r_p2 = st.text_input("Confirm new password", type="password", key="reset_password_2")

                if st.button("Reset Password", use_container_width=True):
                    if not r_uname.strip():
                        st.error("Please enter your username.")
                    elif not user_exists(r_uname.strip()):
                        st.error("No account found with that username.")
                    elif len(r_p1) < 1:
                        st.error("Please enter a new password.")
                    elif r_p1 != r_p2:
                        st.error("Passwords do not match.")
                    else:
                        reset_user_password(r_uname.strip(), r_p1)
                        st.success("Password reset successfully. You can now log in with your new password.")

                if st.button("⬅ Back to Login", key="back_to_login_from_reset"):
                    st.session_state.auth_action = "login"
                    st.rerun()

    st.markdown("<div style='text-align:center; margin-top:15px;'>", unsafe_allow_html=True)
    if st.button("⬅ Back to Homepage"):
        st.session_state.entry_page = "landing"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ==========================================================
# AUTHENTICATED REGION
# ==========================================================
username = st.session_state.username
role = get_user_role(username)

if os.path.exists(BG_PATH):
    st.sidebar.image(BG_PATH, width=120)
st.sidebar.title("ALVAN")
st.sidebar.success(f"👤 {username}")
st.sidebar.info(f"Role : {role.capitalize()}")
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.stream_started = False
    st.session_state.page = "dashboard"
    st.session_state.admin_page = "overview"
    st.session_state.entry_page = "landing"
    st.session_state.login_mode = "user"
    st.session_state.auth_action = "login"
    st.rerun()

# ----------------------------------------------------------
# PAGE 2: DASHBOARD  (role == "user")
# ----------------------------------------------------------
if role == "user" and st.session_state.page == "dashboard":

    st.title("ΛLVΛN — UPI Fraud Detection (Live Demo)")
    st.title("📊 AI Fraud Detection Dashboard")

    st.markdown(
        """
        Welcome to **ALVAN AI**, an intelligent UPI Fraud Detection System
        that continuously monitors transactions using Machine Learning
        and Explainable AI.
        """
    )

    current_time = datetime.now()
    hour = current_time.hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    st.markdown(f"### {greeting}, {username} 👋")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👤 User", username)
    with c2:
        st.metric("🛡 Role", role.capitalize())
    with c3:
        st.metric("📅 Date", current_time.strftime("%d-%m-%Y"))
    with c4:
        st.metric("🕒 Time", current_time.strftime("%I:%M %p"))

    st.markdown("---")

    st.subheader("📈 Your Activity Summary")

    uh = load_user_history(username)
    total_tx = len(uh)
    flagged_tx = int((uh["fraud_flag"] == "⚠️ FRAUD").sum()) if not uh.empty else 0
    safe_tx = total_tx - flagged_tx
    avg_prob = float(uh["fraud_probability"].dropna().astype(float).mean()) if not uh.empty else 0.0

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("💳 Total Transactions", total_tx)
    with s2:
        st.metric("✅ Safe", safe_tx)
    with s3:
        st.metric("⚠️ Flagged as Fraud", flagged_tx)
    with s4:
        st.metric("🎯 Avg Fraud Probability", f"{avg_prob:.2f}%")

    with st.container(border=True):
        if flagged_tx == 0:
            st.success("🛡️ Account Status: No suspicious activity detected on your account.")
        else:
            st.warning(f"⚠️ Account Status: {flagged_tx} transaction(s) flagged for review.")

    st.write("**🕘 Recent Transactions**")
    if not uh.empty:
        recent = uh.sort_values("timestamp", ascending=False).head(5).copy()
        recent_display = recent[["timestamp", "merchant_id", "txn_type", "amount", "fraud_flag"]].copy()
        recent_display["amount"] = recent_display["amount"].apply(
            lambda x: f"₹{float(x):,.2f}" if pd.notna(x) else x
        )
        recent_display.columns = ["Time", "To/From", "Type", "Amount", "Status"]
        st.dataframe(recent_display, use_container_width=True, hide_index=True)
    else:
        st.info("No recent transactions yet.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("➡ Continue to Transaction", use_container_width=True):
            st.session_state.page = "transaction"
            st.rerun()
    with col2:
        if st.button("📷 Scan to Pay", use_container_width=True):
            st.session_state.page = "scan"
            st.rerun()
    with col3:
        if st.button("📜 View Transaction History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

    st.stop()


def get_chatbot_response(query: str, username: str) -> str:
    """
    Rule-based assistant for the dashboard. Answers common questions about
    delays, blocks, and "money deducted but not received" — and when the
    question is personal ("my transaction", "not received", etc.), it looks
    at the user's own most recent transaction to give a specific answer
    instead of a generic one.
    """
    q = query.lower().strip()

    personal_keywords = [
        "my transaction", "my money", "my payment", "i sent", "i received",
        "hasn't arrived", "hasnt arrived", "not received", "didn't receive",
        "didnt receive", "stuck", "pending", "where is my money", "where's my money",
    ]
    if any(k in q for k in personal_keywords):
        hist = load_user_history(username)
        sent = hist[hist["user_id"] == username] if not hist.empty else hist
        if sent is None or sent.empty:
            return "I don't see any transactions on your account yet — send one first and I can explain its status for you."

        last = sent.sort_values("timestamp", ascending=False).iloc[0]
        status = last.get("transaction_status", "UNKNOWN")
        flag = last.get("fraud_flag", "")
        try:
            amt = f"₹{float(last['amount']):,.2f}"
        except Exception:
            amt = str(last.get("amount", ""))

        if status == "BLOCKED":
            return (
                f"Your most recent transaction ({amt} to **{last['merchant_id']}**) was **blocked** by our fraud "
                f"detection system — it was flagged as **{flag}**. That's exactly why the receiver never got "
                "credited: in this simulation, the payment is stopped before any transfer happens once it's "
                "flagged as high risk. No real money left your account. You can see the exact reasons under "
                "'View Detailed Explanation' on the Result page."
            )
        elif status == "APPROVED":
            return (
                f"Your most recent transaction ({amt} to **{last['merchant_id']}**) was **approved** and marked as "
                "simulated success. In this demo, approved payments settle instantly — there's no real waiting "
                "period since no real money moves through the actual UPI network here."
            )
        return f"Your most recent transaction ({amt} to **{last['merchant_id']}**) has status: **{status}**."

    if any(k in q for k in ["why is it taking time", "why is it slow", "how long", "delay", "delayed"]):
        return (
            "In this system, a transaction is either **approved instantly** or **blocked instantly** — there's no "
            "processing wait. If a payment looks stuck, it almost always means it was **blocked** for fraud risk, "
            "not that it's still processing. Check 'View Transaction History' to see the exact status."
        )

    if any(k in q for k in ["block", "why was i blocked", "why blocked", "why did it fail"]):
        return (
            "Transactions get blocked when the fraud model estimates a high probability of fraud — common triggers "
            "are sending to an unverified/unregistered recipient, an amount far above your usual average, or a "
            "recent device/IP change. The exact reasons for any transaction are shown on the 'Detailed AI "
            "Explanation' page."
        )

    if any(k in q for k in ["receiver credited", "not credited", "money deducted", "money debited"]):
        return (
            "This app runs in **simulation mode** — no real money moves through the actual UPI network. If a "
            "transaction is blocked, the receiver is never credited and nothing actually leaves your account, "
            "even though the amount you entered is shown on screen. 'Receiver credited: NO' means the block "
            "happened before any transfer would occur."
        )

    if any(k in q for k in ["fraud", "how do you detect", "how does this work", "how does fraud detection work"]):
        return (
            "ALVAN scores every transaction with a LightGBM model trained on transaction patterns, explains each "
            "decision with SHAP and LIME, and separately scans for fraud rings by analyzing the transaction graph "
            "across all accounts. Any transaction above the fraud probability threshold is blocked automatically."
        )

    if any(k in q for k in ["hi", "hello", "hey"]):
        return "Hi! I'm the ALVAN assistant. Ask me things like 'why was my transaction blocked?' or 'why hasn't the receiver gotten the money?'"

    return (
        "I'm not sure about that one yet — try asking things like *'why was my transaction blocked?'*, "
        "*'why hasn't the receiver gotten the money?'*, or *'how does fraud detection work?'*"
    )


def compute_merchant_risk_score(merchant_id: str, is_registered: bool = True) -> float:
    """
    Looks at every past transaction across ALL users where merchant_id was
    the recipient, and returns the observed fraud rate for that merchant —
    with a moderate default for a merchant with no track record yet, and a
    higher default for a recipient that isn't even a registered/verified
    UPI ID on this platform (a genuinely unknown payee).
    """
    total = 0
    fraud = 0
    for u in registered_users():
        df = load_user_history(u)
        if df is None or df.empty:
            continue
        received = df[df["merchant_id"] == merchant_id]
        total += len(received)
        try:
            fraud += int((received["fraud_flag"] == "⚠️ FRAUD").sum())
        except Exception:
            pass

    if total == 0:
        return 0.20 if is_registered else 0.40  # unverified/unregistered payee starts moderately higher

    rate = fraud / total
    return round(min(1.0, rate + 0.05), 2)  # small smoothing so a single fraud hit doesn't overreact


def compute_user_risk_profile(username: str, recipient: str, amount: float) -> dict:
    """
    Derives all the behavioral/device/velocity/merchant signals automatically
    from the sender's own stored transaction history, this platform's record
    of the recipient, and simulated device/session context — instead of
    asking the sender to self-report them (which a real fraud system never
    does).
    """
    hist = load_user_history(username)
    sent = hist[hist["user_id"] == username] if not hist.empty else hist

    if sent is None or sent.empty:
        user_avg_amount = round(float(amount), 2)
        user_txn_count = 0
        previous_fraud_count = 0
        txn_velocity = 0.0
        known_merchants = set()
    else:
        user_avg_amount = round(float(sent["amount"].astype(float).mean()), 2)
        user_txn_count = int(len(sent))
        raw_fraud_count = int((sent["fraud_flag"] == "⚠️ FRAUD").sum())
        previous_fraud_count = min(raw_fraud_count, 5)
        try:
            ts = pd.to_datetime(sent["timestamp"], errors="coerce")
            recent = sent[ts >= (datetime.now() - pd.Timedelta(minutes=60))]
            txn_velocity = float(len(recent))
        except Exception:
            txn_velocity = 0.0
        known_merchants = set(sent["merchant_id"].dropna().unique().tolist())

    # Account age, straight from the users DB registration timestamp.
    user_age_days = 0
    for r in read_users():
        if r["username"] == username:
            try:
                created = datetime.strptime(r["created"], "%Y-%m-%d %H:%M:%S")
                user_age_days = max(0, (datetime.now() - created).days)
            except Exception:
                user_age_days = 0
            break

    # Deterministic per-user device count (stable across sessions/restarts,
    # unlike a random draw, since a person's device count doesn't change
    # transaction-to-transaction).
    user_device_count = (int(hashlib.md5(username.encode()).hexdigest(), 16) % 4) + 1

    is_registered_recipient = recipient in registered_users()
    is_new_merchant = 0 if recipient in known_merchants else 1
    is_unknown_recipient = 0 if is_registered_recipient else 1

    # No real device/IP telemetry is available in this simulation, so these
    # are drawn from realistic low-probability distributions rather than
    # being entered by the sender.
    ip_change = int(np.random.choice([0, 1], p=[0.92, 0.08]))
    device_change = int(np.random.choice([0, 1], p=[0.92, 0.08]))

    # Simulated geolocation distance — new/unfamiliar merchants are modeled
    # as more likely to be geographically distant, and a genuinely unverified
    # recipient (not even registered on this platform) even more so.
    if is_unknown_recipient:
        geo_distance_km = round(float(np.random.uniform(20, 400)), 2)
    elif is_new_merchant:
        geo_distance_km = round(float(np.random.uniform(5, 150)), 2)
    else:
        geo_distance_km = round(float(np.random.uniform(1, 50)), 2)

    merchant_risk_score = compute_merchant_risk_score(recipient, is_registered=is_registered_recipient)

    return {
        "user_avg_amount": user_avg_amount,
        "user_txn_count": user_txn_count,
        "user_age_days": user_age_days,
        "user_device_count": user_device_count,
        "merchant_risk_score": merchant_risk_score,
        "ip_change": ip_change,
        "device_change": device_change,
        "txn_velocity": txn_velocity,
        "previous_fraud_count": previous_fraud_count,
        "is_new_merchant": is_new_merchant,
        "is_unknown_recipient": is_unknown_recipient,
        "geo_distance_km": geo_distance_km,
    }

# ----------------------------------------------------------
# PAGE 3: TRANSACTION  (role == "user")
# ----------------------------------------------------------
if role == "user" and st.session_state.page == "transaction":

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.subheader("💳 Manual Transaction")
    st.caption(
        "Enter the payment details below — behavioral, device, and merchant "
        "risk signals are computed automatically from account history, the "
        "same way a real fraud detection system works."
    )

    recipients = registered_users()
    if username in recipients:
        recipients.remove(username)

    scanned_recipient = st.session_state.pop("scanned_recipient", None)

    default_unknown = bool(scanned_recipient and scanned_recipient not in recipients)

    if scanned_recipient:
        st.info(f"📷 Recipient pre-filled from your scanned QR code: **{scanned_recipient}**")

    send_to_unknown = st.checkbox(
        "Send to someone not in ALVAN (unknown / external recipient)",
        value=default_unknown
    )

    with st.form("manual_txn"):
        if send_to_unknown:
            recipient = st.text_input(
                "Recipient name or phone number",
                value=scanned_recipient if scanned_recipient else ""
            )
        else:
            default_index = recipients.index(scanned_recipient) if (scanned_recipient and scanned_recipient in recipients) else 0
            recipient = st.selectbox(
                "Send To",
                recipients if recipients else ["merchant_1"],
                index=default_index
            )
        txn_type = st.selectbox("Txn Type", ["send", "receive", "withdraw", "upi_collect"])
        upi_app = st.selectbox("UPI App", ["GPay", "PhonePe", "Paytm", "AmazonPay", "BHIM"])
        amount = st.number_input("Amount (₹)", min_value=1.00, max_value=5000000.00, value=1500.00, step=0.01)

        submitted = st.form_submit_button("Send Transaction")

    if submitted:
        errors = []
        if amount <= 0:
            errors.append("Amount must be greater than 0.")
        if send_to_unknown and not recipient.strip():
            errors.append("Please enter the recipient's name or phone number.")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        recipient = recipient.strip()

        risk_profile = compute_user_risk_profile(username, recipient, amount)

        txn = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": username,
            "merchant_id": recipient,
            "txn_type": txn_type,
            "upi_app": upi_app,
            "amount": round(float(amount), 2),
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            **risk_profile,
        }

        # --------------------------------------------------
        # SEND TRANSACTION TO FLASK FRAUD API
        # --------------------------------------------------
        api_result = predict_transaction_api(txn)

        # Do not continue if the fraud decision could not be made.
        if not api_result.get("success", False):
            st.error(
                "❌ Transaction could not be approved because "
                "the fraud detection service was unavailable."
            )
            st.warning(
                api_result.get(
                    "error",
                    "Fraud prediction could not be completed."
                )
            )
            st.stop()

        # --------------------------------------------------
        # STORE THE COMPLETE API DECISION
        # --------------------------------------------------
        prob = float(api_result.get("fraud_probability", 0.0))

        risk_level = api_result.get("risk_level", "UNKNOWN")
        transaction_status = api_result.get(
            "transaction_status", "UNKNOWN"
        )
        payment_status = api_result.get(
            "payment_status", "UNKNOWN"
        )
        receiver_credited = bool(
            api_result.get("receiver_credited", False)
        )
        api_message = api_result.get(
            "message",
            "Transaction processed."
        )

        txn["fraud_probability"] = round(prob * 100.0, 2)
        txn["fraud_flag"] = (
            "⚠️ FRAUD"
            if int(api_result.get("is_fraud", 0)) == 1
            else "✅ SAFE"
        )
        txn["risk_level"] = risk_level
        txn["transaction_status"] = transaction_status
        txn["payment_status"] = payment_status
        txn["receiver_credited"] = receiver_credited
        txn["api_message"] = api_message

        # --------------------------------------------------
        # SAVE AUDIT / USER HISTORY
        # --------------------------------------------------
        append_fraud_alert(txn)

        analysis = explain_transaction(
            txn,
            txn["fraud_probability"]
        )

        append_user_history(username, txn)

        if recipient in registered_users():
            append_user_history(recipient, txn)

        # --------------------------------------------------
        # LIVE STREAM RECORD
        # --------------------------------------------------
        st.session_state.GLOBAL_STREAM.insert(0, txn)

        if len(st.session_state.GLOBAL_STREAM) > MAX_STREAM_LEN:
            st.session_state.GLOBAL_STREAM = (
                st.session_state.GLOBAL_STREAM[:MAX_STREAM_LEN]
            )

        # --------------------------------------------------
        # SAVE RESULT FOR RESULT PAGE
        # --------------------------------------------------
        st.session_state.last_txn = txn
        st.session_state.last_analysis = analysis
        st.session_state.last_risk_profile = risk_profile
        st.session_state.page = "result"
        st.rerun()

    st.stop()

# ----------------------------------------------------------
# PAGE 4: RESULT  (role == "user")
# ----------------------------------------------------------
if role == "user" and st.session_state.page == "result":

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    txn = st.session_state.get("last_txn")
    analysis = st.session_state.get("last_analysis")

    if txn is None or analysis is None:
        st.warning("No transaction result to show. Please send a transaction first.")
        st.stop()

    st.markdown("## 🤖 AI Fraud Analysis Report")

    with st.container(border=True):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.write("**To/From**")
            st.write(txn["merchant_id"])
        with d2:
            st.write("**Type**")
            st.write(txn["txn_type"].title())
        with d3:
            st.write("**App**")
            st.write(txn["upi_app"])
        with d4:
            st.write("**Time**")
            st.write(txn["timestamp"])

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("💰 Amount", f"₹{txn['amount']:,.2f}")
    with colB:
        st.metric("🎯 Fraud Probability", f"{analysis['probability']:.2f}%")
    with colC:
        st.metric("🚩 Verdict", txn["fraud_flag"])

    st.progress(min(int(analysis["probability"]), 100))

    risk_profile = st.session_state.get("last_risk_profile")
    if risk_profile:
        with st.expander("🧮 Auto-computed risk signals (not entered by you)"):
            st.caption(
                "These signals were derived automatically from your account history, "
                "this merchant's track record, and simulated device/session context — "
                "none of them were entered manually."
            )
            rp1, rp2, rp3 = st.columns(3)
            with rp1:
                st.metric("Your avg. amount", f"₹{risk_profile['user_avg_amount']:,.2f}")
                st.metric("Your txn count", risk_profile["user_txn_count"])
                st.metric("Account age (days)", risk_profile["user_age_days"])
            with rp2:
                st.metric("Merchant risk score", f"{risk_profile['merchant_risk_score']:.2f}")
                st.metric("New merchant?", "Yes" if risk_profile["is_new_merchant"] else "No")
                st.metric("Verified recipient?", "No" if risk_profile.get("is_unknown_recipient") else "Yes")
            with rp3:
                st.metric("Devices on account", risk_profile["user_device_count"])
                st.metric("IP changed", "Yes" if risk_profile["ip_change"] else "No")
                st.metric("Device changed", "Yes" if risk_profile["device_change"] else "No")
                st.metric("Prior fraud count", risk_profile["previous_fraud_count"])

    # ------------------------------------------------------
    # API TRANSACTION DECISION
    # ------------------------------------------------------
    st.markdown("### 🔐 Transaction Decision")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Risk Level",
            txn.get("risk_level", "UNKNOWN")
        )

    with d2:
        st.metric(
            "Transaction Status",
            txn.get("transaction_status", "UNKNOWN")
        )

    with d3:
        st.metric(
            "Payment Status",
            txn.get("payment_status", "UNKNOWN")
        )

    if txn.get("transaction_status") == "BLOCKED":

        st.error("🚫 TRANSACTION BLOCKED")

        st.warning(
            "The payment was NOT executed because high fraud "
            "risk was detected."
        )

        st.info(
            "👤 Receiver credited: NO"
        )

    elif txn.get("transaction_status") == "APPROVED":

        st.success("✅ TRANSACTION APPROVED")

        st.success(
            "Payment status: SIMULATED SUCCESS"
        )

        st.info(
            "👤 Receiver credited: YES (simulation only)"
        )

    st.caption(
        "Simulation mode: this prototype does not transfer real money "
        "through the UPI network."
    )

    if txn.get("api_message"):
        st.write(
            f"**System message:** {txn['api_message']}"
        )

    is_blocked = txn.get("transaction_status") == "BLOCKED"

    if is_blocked and analysis["probability"] >= 90:
        risk_bg, risk_border = "#4a1414", "#e74c3c"
        risk_label = "🔴 HIGH RISK"
        recommendation = "🚫 Blocked automatically — escalate for manual investigation immediately."
    elif is_blocked:
        risk_bg, risk_border = "#4a3814", "#f39c12"
        risk_label = "🟠 ELEVATED RISK (Blocked)"
        recommendation = "🚫 Blocked pending manual review — do not retry until the account is verified."
    elif analysis["probability"] >= 30:
        risk_bg, risk_border = "#4a3814", "#f1c40f"
        risk_label = "🟡 LOW-MODERATE RISK"
        recommendation = "✅ Approved, but flag this account for closer monitoring going forward."
    else:
        risk_bg, risk_border = "#123d1c", "#2ecc71"
        risk_label = "🟢 LOW RISK"
        recommendation = "✅ Transaction appears legitimate."

    st.markdown(
        f"""
        <div style='
            background:{risk_bg};
            border:1px solid {risk_border};
            border-radius:12px;
            padding:22px;
            margin-top:10px;
        '>
            <h3 style='color:#FFFFFF; margin:0;'>{risk_label}</h3>
            <p style='color:#EAEAEA; margin-top:10px; font-size:16px; margin-bottom:0;'>{recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 View Detailed Explanation", use_container_width=True):
        st.session_state.page = "explanation"
        st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡ View Transaction History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()
    with col2:
        if st.button("🔁 Send Another Transaction", use_container_width=True):
            st.session_state.page = "transaction"
            st.rerun()

    st.markdown("---")

    chat_toggle_col, _ = st.columns([1, 4])
    with chat_toggle_col:
        if st.button("💬 Ask ALVAN", use_container_width=True, key="chat_widget_toggle"):
            st.session_state.show_chat_widget = not st.session_state.show_chat_widget
            st.rerun()

    if st.session_state.show_chat_widget:
        with st.container(border=True):
            st.caption(
                "Ask why this transaction was blocked, delayed, or why the receiver hasn't gotten "
                "the money — the assistant looks at your own transaction history to answer."
            )
            for chat_role, chat_msg in st.session_state.chat_history[-12:]:
                with st.chat_message(chat_role):
                    st.write(chat_msg)

        user_query = st.chat_input("e.g. Why hasn't the receiver gotten my money?")
        if user_query:
            st.session_state.chat_history.append(("user", user_query))
            reply = get_chatbot_response(user_query, username)
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()

    st.stop()

# ----------------------------------------------------------
# PAGE 4b: EXPLANATION  (role == "user", separate page for
# "why did the AI make this prediction" + SHAP)
# ----------------------------------------------------------
if role == "user" and st.session_state.page == "explanation":

    if st.button("⬅ Back to Result"):
        st.session_state.page = "result"
        st.rerun()

    txn = st.session_state.get("last_txn")
    analysis = st.session_state.get("last_analysis")

    if txn is None or analysis is None:
        st.warning("No transaction result to show. Please send a transaction first.")
        st.stop()

    st.markdown("## 🔍 Detailed AI Explanation")
    st.caption(f"Explanation for transaction to **{txn['merchant_id']}** — ₹{txn['amount']:,.2f} on {txn['timestamp']}")
    st.markdown("---")

    with st.container(border=True):
        st.write("### 📋 Why did the AI make this prediction?")
        for reason in analysis["reasons"]:
            st.write("•", reason)

    st.markdown("<br>", unsafe_allow_html=True)

    shap_result = get_shap_explanation(txn)

    with st.container(border=True):
        st.write("### 🧠 SHAP Feature Contributions")

        if shap_result is not None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            labels = [pretty_feature_name(f) for f in shap_result.index]
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_result.values]
            bars = ax.barh(labels[::-1], shap_result.values[::-1], color=colors[::-1])

            for bar, val in zip(bars, shap_result.values[::-1]):
                x = bar.get_width()
                align = "left" if x >= 0 else "right"
                offset = 0.01 * (max(abs(shap_result.values)) or 1)
                ax.text(
                    x + (offset if x >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f"{x:+.3f}",
                    va="center",
                    ha=align,
                    fontsize=9,
                    color="#FFFFFF",
                )

            ax.axvline(x=0, color="#8FA8C7", linewidth=0.8)
            ax.set_title("Top Factors Influencing This Prediction", fontsize=13)
            ax.set_xlabel("Impact on Fraud Probability (SHAP value)")
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("Red = pushes toward FRAUD, Green = pushes toward SAFE.")

            top_factors = shap_result.head(2)
            factor_phrases = []
            for name, val in top_factors.items():
                direction = "increased" if val > 0 else "decreased"
                factor_phrases.append(f"**{pretty_feature_name(name)}** ({direction} risk, impact {val:+.3f})")

            if factor_phrases:
                summary_sentence = "This prediction was driven mainly by " + " and ".join(factor_phrases) + "."
                st.info(f"🗣️ **In plain terms:** {summary_sentence}")
        else:
            st.warning("SHAP explanation is unavailable for this transaction.")

    st.markdown("<br>", unsafe_allow_html=True)

    lime_result = get_lime_explanation(txn)

    with st.container(border=True):
        st.write("### 🍋 LIME Feature Contributions")
        st.caption(
            "An independent explanation method, computed separately from SHAP, "
            "for the same LightGBM prediction — useful for cross-checking that "
            "both methods agree on what drove the decision."
        )

        if lime_result:
            lime_labels = [pair[0] for pair in lime_result][::-1]
            lime_values = [pair[1] for pair in lime_result][::-1]
            lime_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in lime_values]

            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            bars = ax3.barh(lime_labels, lime_values, color=lime_colors)

            for bar, val in zip(bars, lime_values):
                x = bar.get_width()
                align = "left" if x >= 0 else "right"
                offset = 0.01 * (max(abs(v) for v in lime_values) or 1)
                ax3.text(
                    x + (offset if x >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f"{x:+.3f}",
                    va="center",
                    ha=align,
                    fontsize=9,
                    color="#FFFFFF",
                )

            ax3.axvline(x=0, color="#8FA8C7", linewidth=0.8)
            ax3.set_title("Top Factors Influencing This Prediction (LIME)", fontsize=13)
            ax3.set_xlabel("Contribution to Fraud Probability")
            fig3.tight_layout()
            st.pyplot(fig3)
            st.caption("Red = pushes toward FRAUD, Green = pushes toward SAFE.")
        else:
            st.warning("LIME explanation is unavailable for this transaction.")

    st.markdown("<br>", unsafe_allow_html=True)

    recommendation = get_recommendation(txn, analysis, shap_result, lime_result)

    with st.container(border=True):
        st.write("### 🧭 Recommended Action")

        st.markdown(
            f"""
            <div style='
                background:#0F2A4A;
                border:2px solid {recommendation["color"]};
                border-radius:12px;
                padding:22px;
                margin-top:5px;
            '>
                <h2 style='color:{recommendation["color"]}; margin:0;'>{recommendation["action"]}</h2>
                <p style='color:#DDE6F5; margin-top:6px; margin-bottom:0;'>
                    <strong>Confidence:</strong> {recommendation["confidence"]}
                </p>
                <p style='color:#EAEAEA; margin-top:12px; font-size:15px; margin-bottom:0;'>
                    {recommendation["justification"]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if recommendation["methods_agree"]:
            st.caption("✅ SHAP and LIME agree on the top risk driver for this transaction — increases confidence in this recommendation.")
        else:
            st.caption("ℹ️ SHAP and LIME highlight different top factors — recommendation is based primarily on the overall fraud probability and rule-based signals.")

    st.markdown("---")
    if st.button("➡ Back to Result Summary", use_container_width=True):
        st.session_state.page = "result"
        st.rerun()

    st.stop()

# ----------------------------------------------------------
# PAGE 5: HISTORY  (role == "user")
# ----------------------------------------------------------
if role == "user" and st.session_state.page == "history":

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.markdown("---")
    st.subheader("📁 Your Transaction History")

    uh = load_user_history(username)

    if uh.empty:
        st.info("No transactions yet.")
    else:
        uh_display = uh.copy()
        try:
            uh_display["amount"] = uh_display["amount"].apply(lambda x: f"₹{float(x):,.2f}" if pd.notna(x) else x)
            uh_display["fraud_probability"] = uh_display["fraud_probability"].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else x)
        except Exception:
            pass
        st.dataframe(uh_display.sort_values(by="timestamp", ascending=False).reset_index(drop=True), use_container_width=True, height=450)

    st.stop()

# ----------------------------------------------------------
# PAGE 3b: SCAN TO PAY  (role == "user")
# ----------------------------------------------------------
if role == "user" and st.session_state.page == "scan":

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    st.subheader("📷 Scan to Pay — Merchant Safety Check")
    st.caption(
        "Scan a UPI QR code to see who you're actually paying before you send money. "
        "This is exactly how real UPI scam prevention works — most scan-and-pay fraud "
        "happens because people never see who's really on the other end."
    )

    scan_mode = st.radio("Choose scan method", ["📁 Upload QR Image", "📷 Live Camera Scan"], horizontal=True)

    scanned_data = None

    if scan_mode == "📁 Upload QR Image":
        uploaded_file = st.file_uploader("Upload a QR code image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            scanned_data = decode_qr_from_image(uploaded_file)
            if scanned_data is None:
                st.error("No QR code could be detected in that image. Try a clearer photo.")

    else:
        st.info(
            "Clicking below opens a separate camera window on this computer. Point a QR "
            "code at your webcam — it closes automatically once detected. Press ESC to cancel."
        )
        if st.button("🎥 Start Camera Scan"):
            with st.spinner("Opening camera... point a QR code at your webcam"):
                scanned_data = scan_qr_live()
            if scanned_data is None:
                st.warning("No QR code detected (camera closed without a scan).")

    if scanned_data:
        parsed = parse_upi_qr(scanned_data)
        merchant_identifier = parsed["vpa"] or parsed["raw"]

        st.markdown("---")
        st.success(f"✅ QR Code Scanned: **{merchant_identifier}**")
        if parsed["name"]:
            st.write(f"**Merchant Name:** {parsed['name']}")
        if parsed["amount"]:
            st.write(f"**Requested Amount:** ₹{parsed['amount']}")

        trust = check_merchant_trust(merchant_identifier)

        st.markdown("---")
        st.write("### 🔍 Preliminary Safety Check")
        st.caption("A quick heads-up based on this platform's history — the full fraud check runs on the Transaction page.")

        if trust["verdict"] == "new":
            st.warning("🟡 **New / Unverified Merchant** — no transaction history found on this platform. Proceed with caution.")
        elif trust["verdict"] == "risky":
            st.error(f"🔴 **High Risk Merchant** — {trust['fraud_count']} of {trust['total_transactions']} past transaction(s) with this merchant were flagged as fraud.")
        else:
            st.success(f"✅ **Known Merchant** — {trust['total_transactions']} prior transaction(s) on record, no fraud history.")

        st.markdown("---")
        if st.button("➡ Continue to Transaction with this recipient", use_container_width=True):
            st.session_state.scanned_recipient = merchant_identifier
            st.session_state.page = "transaction"
            st.rerun()

    st.stop()

# ==========================================================
# ADMIN VIEW — role == "admin" only, split into its own pages
# ==========================================================
if role != "admin":
    # Safety net: any role that is neither "user" nor "admin" should never
    # see admin content. Every "user" branch above already ends in st.stop(),
    # so reaching here with role == "user" is impossible — this is just a guard.
    st.error("Access denied.")
    st.stop()

def build_transaction_graph():
    """
    Builds an undirected graph of every sender <-> recipient relationship
    seen across all users' transaction history. Each edge tracks how many
    transactions passed between that pair and how many were flagged fraud.
    De-duplicates transactions that are stored in both the sender's and
    recipient's own history files.
    """
    seen = set()
    edge_data = {}

    for u in registered_users():
        df = load_user_history(u)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            uid = row.get("user_id")
            mid = row.get("merchant_id")
            ts = row.get("timestamp")
            if pd.isna(uid) or pd.isna(mid):
                continue

            dedup_key = (str(uid), str(mid), str(ts), str(row.get("amount")))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            is_fraud = row.get("fraud_flag") == "⚠️ FRAUD"
            key = tuple(sorted([str(uid), str(mid)]))
            if key not in edge_data:
                edge_data[key] = {"weight": 0, "fraud_count": 0}
            edge_data[key]["weight"] += 1
            if is_fraud:
                edge_data[key]["fraud_count"] += 1

    G = nx.Graph()
    for (a, b), data in edge_data.items():
        G.add_edge(a, b, weight=data["weight"], fraud_count=data["fraud_count"])

    return G


def detect_fraud_clusters(G, fraud_rate_threshold=0.3, min_component_size=2):
    """
    Finds connected components (clusters of users/merchants linked by
    transactions) whose combined fraud rate exceeds a threshold — these
    are candidate fraud rings, as opposed to individually-flagged
    transactions which only look at one transaction in isolation.
    """
    clusters = []
    for component in nx.connected_components(G):
        if len(component) < min_component_size:
            continue

        subgraph = G.subgraph(component)
        total_txns = sum(d["weight"] for _, _, d in subgraph.edges(data=True))
        fraud_txns = sum(d["fraud_count"] for _, _, d in subgraph.edges(data=True))

        if total_txns == 0:
            continue

        fraud_rate = fraud_txns / total_txns
        clusters.append({
            "members": sorted(component),
            "size": len(component),
            "total_txns": total_txns,
            "fraud_txns": fraud_txns,
            "fraud_rate": fraud_rate,
            "suspicious": fraud_rate >= fraud_rate_threshold and fraud_txns >= 2,
        })

    clusters.sort(key=lambda c: c["fraud_rate"], reverse=True)
    return clusters


ADMIN_PAGES = ["overview", "users", "alerts", "performance", "comparison", "ring", "live"]
if st.session_state.admin_page not in ADMIN_PAGES:
    st.session_state.admin_page = "overview"

st.title("ΛLVΛN — Admin Control Center")

# ---- Admin top navigation ----
nav_labels = {
    "overview": "🏠 Overview",
    "users": "👥 All Users",
    "alerts": "🚨 Fraud Alerts",
    "performance": "📊 Model Performance",
    "comparison": "🏆 Model Comparison",
    "ring": "🕸️ Fraud Rings",
    "live": "📡 Live Stream",
}
nav_cols = st.columns(len(ADMIN_PAGES))
for i, key in enumerate(ADMIN_PAGES):
    with nav_cols[i]:
        if st.button(nav_labels[key], use_container_width=True,
                      type="primary" if st.session_state.admin_page == key else "secondary"):
            st.session_state.admin_page = key
            st.rerun()

st.markdown("---")

users = registered_users()

# ----------------------------------------------------------
# ADMIN PAGE: OVERVIEW
# ----------------------------------------------------------
if st.session_state.admin_page == "overview":

    st.subheader("🏠 System Overview")

    hist_tx, hist_flagged, hist_probs = 0, 0, []
    for u in users:
        df = load_user_history(u)
        if df is None or df.empty:
            continue
        hist_tx += len(df)
        try:
            hist_flagged += int((df["fraud_flag"] == "⚠️ FRAUD").sum())
        except Exception:
            pass
        try:
            hist_probs.extend([float(x) for x in df["fraud_probability"].dropna().tolist()])
        except Exception:
            pass

    avg_prob = float(np.mean(hist_probs)) if hist_probs else 0.0

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        st.metric("👥 Total Users", len(users))
    with o2:
        st.metric("💳 Total Transactions", hist_tx)
    with o3:
        st.metric("⚠️ Flagged as Fraud", hist_flagged)
    with o4:
        st.metric("🎯 Avg Fraud Probability", f"{avg_prob:.2f}%")

    st.info("Use the navigation bar above to view all users, model performance, model comparison, or the live transaction stream.")

# ----------------------------------------------------------
# ADMIN PAGE: ALL USERS
# ----------------------------------------------------------
elif st.session_state.admin_page == "users":

    st.subheader("👥 All Users & Their Transaction Histories")

    users_df = pd.DataFrame([{"username": u, "role": get_user_role(u)} for u in users])
    st.table(users_df)

    selected = st.selectbox("Select a user to view history", ["--choose--"] + users)
    if selected != "--choose--":
        hist = load_user_history(selected)
        if hist.empty:
            st.info("No transactions for this user.")
        else:
            hist_view = hist.copy()
            try:
                hist_view["amount"] = hist_view["amount"].apply(lambda x: f"₹{float(x):,.2f}")
                hist_view["fraud_probability"] = hist_view["fraud_probability"].apply(lambda x: f"{float(x):.2f}%")
            except Exception:
                pass
            st.dataframe(hist_view.sort_values("timestamp", ascending=False), height=400)

# ----------------------------------------------------------
# ADMIN PAGE: FRAUD ALERTS  (audit log — only FRAUD-flagged transactions)
# ----------------------------------------------------------
elif st.session_state.admin_page == "alerts":

    st.subheader("🚨 Fraud Alert Log")
    st.caption(
        "Every transaction flagged as FRAUD is automatically logged here for review — "
        "a separate audit trail from normal transaction history."
    )

    alerts_df = load_fraud_alerts()

    total_alerts = len(alerts_df)
    unreviewed = int((alerts_df["reviewed"] == "No").sum()) if not alerts_df.empty else 0
    reviewed = total_alerts - unreviewed

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Total Fraud Alerts", total_alerts)
    with a2:
        st.metric("⏳ Pending Review", unreviewed)
    with a3:
        st.metric("✅ Reviewed", reviewed)

    st.markdown("---")

    filter_choice = st.radio("Filter", ["All", "Pending Review", "Reviewed"], horizontal=True)

    if alerts_df.empty:
        st.info("No fraud alerts have been logged yet. Alerts appear here automatically when a transaction is flagged as FRAUD.")
    else:
        if filter_choice == "Pending Review":
            filtered = alerts_df[alerts_df["reviewed"] == "No"]
        elif filter_choice == "Reviewed":
            filtered = alerts_df[alerts_df["reviewed"] == "Yes"]
        else:
            filtered = alerts_df

        filtered = filtered.sort_values("timestamp", ascending=False)

        if filtered.empty:
            st.info("No fraud alerts match this filter.")
        else:
            st.write(f"**Showing {min(len(filtered), 30)} of {len(filtered)} alert(s):**")

            for _, row in filtered.head(30).iterrows():
                with st.container(border=True):
                    c1, c2, c3, c4, c5 = st.columns([2, 2.2, 1.5, 1.5, 1.5])
                    with c1:
                        st.write("**Time**")
                        st.write(row["timestamp"])
                    with c2:
                        st.write("**User → Merchant**")
                        st.write(f"{row['user_id']} → {row['merchant_id']}")
                    with c3:
                        st.write("**Amount**")
                        st.write(f"₹{float(row['amount']):,.2f}")
                    with c4:
                        st.write("**Probability**")
                        st.write(f"{float(row['fraud_probability']):.2f}%")
                    with c5:
                        if row["reviewed"] == "No":
                            if st.button("Mark Reviewed", key=f"review_{row['alert_id']}"):
                                mark_alert_reviewed(row["alert_id"])
                                st.rerun()
                        else:
                            st.success("✅ Reviewed")

            if len(filtered) > 30:
                st.caption(f"Showing the latest 30 of {len(filtered)} matching alerts. Use the filter above to narrow results.")

# ----------------------------------------------------------
# ADMIN PAGE: MODEL PERFORMANCE
# ----------------------------------------------------------
elif st.session_state.admin_page == "performance":

    st.subheader("📊 Model Performance")
    try:
        y_val = np.load(os.path.join(BASE_DIR, "models", "y_val.npy"))
        y_pred = np.load(os.path.join(BASE_DIR, "models", "y_pred.npy"))
        y_proba = np.load(os.path.join(BASE_DIR, "models", "y_proba.npy"))

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Fraud"])
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            st.pyplot(fig)

        with col2:
            st.write("**Precision-Recall vs Threshold**")
            precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
            fig2, ax2 = plt.subplots()
            ax2.plot(thresholds, precision[:-1], label="Precision", color="#4FC3F7")
            ax2.plot(thresholds, recall[:-1], label="Recall", color="#FFB74D")
            ax2.axvline(x=0.5, color="#E57373", linestyle="--", label="Current threshold (0.5)")
            ax2.set_xlabel("Threshold")
            ax2.set_ylabel("Score")
            ax2.legend()
            st.pyplot(fig2)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Accuracy", f"{accuracy_score(y_val, y_pred)*100:.2f}%")
        with m2:
            st.metric("Precision", f"{precision_score(y_val, y_pred)*100:.2f}%")
        with m3:
            st.metric("Recall", f"{recall_score(y_val, y_pred)*100:.2f}%")
        with m4:
            st.metric("F1 Score", f"{f1_score(y_val, y_pred)*100:.2f}%")
        with m5:
            st.metric("ROC AUC", f"{roc_auc_score(y_val, y_proba)*100:.2f}%")

        fi_path = os.path.join(BASE_DIR, "models", "feature_importance.png")
        if os.path.exists(fi_path):
            st.markdown("---")
            st.image(fi_path, caption="Top 20 Feature Importances (LightGBM, gain)", use_container_width=True)

    except FileNotFoundError:
        st.warning("Run train.py first to generate model performance data.")

# ----------------------------------------------------------
# ADMIN PAGE: MODEL COMPARISON
# ----------------------------------------------------------
elif st.session_state.admin_page == "comparison":

    st.subheader("🏆 Model Comparison")
    try:
        comparison_df = pd.read_csv(os.path.join(BASE_DIR, "models", "model_comparison.csv"))

        display_df = comparison_df.copy()
        for col in ["AUC", "Precision", "Recall", "F1"]:
            if col in display_df.columns:
                display_df[col] = (display_df[col].astype(float) * 100).map(lambda x: f"{x:.2f}%")
        st.table(display_df.set_index("Model"))

        best_model = comparison_df.sort_values("AUC", ascending=False).iloc[0]
        st.success(f"🥇 Best performing model: **{best_model['Model']}** (AUC = {best_model['AUC']*100:.2f}%) — currently deployed in this app.")

        st.markdown("---")
        roc_path = os.path.join(BASE_DIR, "models", "roc_comparison.png")
        cal_path = os.path.join(BASE_DIR, "models", "calibration_curve.png")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curve — Model Comparison", use_container_width=True)
        with chart_col2:
            if os.path.exists(cal_path):
                st.image(cal_path, caption="Calibration Curve — LightGBM", use_container_width=True)

        st.markdown("---")
        st.write("### 📈 Real Performance Gains")
        st.caption("Computed live from your actual training results — not a static image.")

        try:
            lgbm_row = comparison_df[comparison_df["Model"] == "LightGBM"].iloc[0]
            rf_row = comparison_df[comparison_df["Model"] == "Random Forest"].iloc[0]
            lr_row = comparison_df[comparison_df["Model"] == "Logistic Regression"].iloc[0]

            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric(
                    "AUC vs Random Forest",
                    f"{lgbm_row['AUC']*100:.2f}%",
                    f"+{(lgbm_row['AUC']-rf_row['AUC'])*100:.2f} pts",
                )
            with g2:
                st.metric(
                    "Precision vs Random Forest",
                    f"{lgbm_row['Precision']*100:.2f}%",
                    f"+{(lgbm_row['Precision']-rf_row['Precision'])*100:.2f} pts",
                )
            with g3:
                st.metric(
                    "Recall vs Random Forest",
                    f"{lgbm_row['Recall']*100:.2f}%",
                    f"+{(lgbm_row['Recall']-rf_row['Recall'])*100:.2f} pts",
                )
            with g4:
                st.metric(
                    "F1 vs Random Forest",
                    f"{lgbm_row['F1']*100:.2f}%",
                    f"+{(lgbm_row['F1']-rf_row['F1'])*100:.2f} pts",
                )

            st.info(
                f"Compared to the simplest baseline (Logistic Regression), **LightGBM improves "
                f"AUC by {(lgbm_row['AUC']-lr_row['AUC'])*100:.2f} percentage points** and "
                f"**Recall by {(lgbm_row['Recall']-lr_row['Recall'])*100:.2f} percentage points** — "
                f"meaning it catches significantly more real fraud cases than a basic linear model."
            )
        except (KeyError, IndexError):
            st.warning("Could not compute comparison deltas — check that model_comparison.csv has all three model rows.")

    except FileNotFoundError:
        st.warning("Run the updated train.py first to generate model comparison data.")

# ----------------------------------------------------------
# ADMIN PAGE: FRAUD RING DETECTOR  (graph-based cluster analysis)
# ----------------------------------------------------------
elif st.session_state.admin_page == "ring":

    st.subheader("🕸️ Fraud Ring Detector")
    st.caption(
        "Instead of scoring one transaction at a time, this builds a graph of every "
        "sender ↔ recipient relationship across the whole platform and looks for "
        "connected clusters with an abnormally high combined fraud rate — a pattern "
        "individual transaction scoring alone can't see."
    )

    G = build_transaction_graph()

    if G.number_of_edges() == 0:
        st.info("Not enough transaction data yet to build a relationship graph. Send a few transactions first.")
    else:
        clusters = detect_fraud_clusters(G)
        suspicious = [c for c in clusters if c["suspicious"]]

        g1, g2, g3 = st.columns(3)
        with g1:
            st.metric("Accounts in graph", G.number_of_nodes())
        with g2:
            st.metric("Relationships tracked", G.number_of_edges())
        with g3:
            st.metric("🚨 Suspicious clusters", len(suspicious))

        st.markdown("---")

        # ------------------------------------------------------
        # GRAPH VISUALIZATION
        # ------------------------------------------------------
        st.write("### 🔗 Transaction Relationship Graph")

        suspicious_members = set()
        for c in suspicious:
            suspicious_members.update(c["members"])

        try:
            fig, ax = plt.subplots(figsize=(10, 7))
            pos = nx.spring_layout(G, seed=42, k=0.6)

            node_colors = ["#e74c3c" if n in suspicious_members else "#4FC3F7" for n in G.nodes()]
            edge_colors = [
                "#e74c3c" if G[u][v]["fraud_count"] > 0 else "#3a5a80"
                for u, v in G.edges()
            ]
            edge_widths = [1 + G[u][v]["weight"] * 0.5 for u, v in G.edges()]

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.6)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=400, edgecolors="#FFFFFF", linewidths=0.8)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="#FFFFFF")

            ax.set_title("Red nodes = part of a suspicious fraud cluster", fontsize=12)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render graph: {e}")

        st.markdown("---")

        # ------------------------------------------------------
        # CLUSTER TABLE
        # ------------------------------------------------------
        st.write("### 📋 Detected Clusters")

        if not clusters:
            st.info("No multi-account clusters detected yet.")
        else:
            for c in clusters[:20]:
                border_color = "#e74c3c" if c["suspicious"] else "#3a5a80"
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style='border-left: 4px solid {border_color}; padding-left: 12px;'>
                        """,
                        unsafe_allow_html=True,
                    )
                    cc1, cc2, cc3, cc4 = st.columns([2, 1, 1, 1])
                    with cc1:
                        st.write("**Members**")
                        st.write(", ".join(c["members"][:6]) + (" ..." if len(c["members"]) > 6 else ""))
                    with cc2:
                        st.write("**Total Txns**")
                        st.write(c["total_txns"])
                    with cc3:
                        st.write("**Fraud Txns**")
                        st.write(c["fraud_txns"])
                    with cc4:
                        st.write("**Fraud Rate**")
                        st.write(f"{c['fraud_rate']*100:.1f}%")
                    if c["suspicious"]:
                        st.error("🚨 Flagged as a suspicious cluster — recommend investigating this group as a potential fraud ring.")
                    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# ADMIN PAGE: LIVE STREAM  (must be the last page — infinite loop)
# ----------------------------------------------------------
elif st.session_state.admin_page == "live":

    st.subheader("📡 Live Transaction Stream (updates every 5s)")

    if not st.session_state.stream_started:
        st.session_state.stream_started = True
        st.rerun()

    live_placeholder = st.empty()

    while True:
        try:
            tx = random_stream_txn()
            prob = predict_prob(tx)
            tx["fraud_probability"] = round(prob * 100.0, 2)
            tx["fraud_flag"] = "⚠️ FRAUD" if prob >= 0.5 else "✅ SAFE"

            st.session_state.GLOBAL_STREAM.insert(0, tx)
            if len(st.session_state.GLOBAL_STREAM) > MAX_STREAM_LEN:
                st.session_state.GLOBAL_STREAM = st.session_state.GLOBAL_STREAM[:MAX_STREAM_LEN]

            with live_placeholder:
                lines = []
                for r in st.session_state.GLOBAL_STREAM[:60]:
                    amt = r.get("amount", "")
                    p = r.get("fraud_probability", "")
                    flag = r.get("fraud_flag", "")
                    ts = r.get("timestamp", "")
                    uid = r.get("user_id", "")
                    mid = r.get("merchant_id", "")
                    lines.append(f"{ts} | {uid} → {mid} | ₹{amt} | {flag} ({p}%)")
                st.code("\n".join(lines), language="bash")

        except Exception as e:
            with live_placeholder:
                st.error(f"Stream error: {e}")

        time.sleep(STREAM_INTERVAL)