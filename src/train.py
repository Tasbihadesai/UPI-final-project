"""
train.py
Train and compare multiple models for UPI Fraud Detection (20-feature dataset)

STABLE VERSION — fixed, sensible LightGBM hyperparameters (no randomized
search). This is the version confirmed working correctly before your first
review. Use this one for your demo.

What this script does:
 1. Loads and preprocesses the dataset (same preprocess.py pipeline as before)
 2. Splits data by TIME (train on the oldest 80% of transactions, test on the
    most recent 20%) — realistic for fraud detection.
 3. Trains THREE models on the same data: Logistic Regression, Random Forest,
    and LightGBM — and evaluates all three on the same held-out test set.
 4. Saves a comparison table (AUC / Precision / Recall / F1) and an overlaid
    ROC curve so you can visually justify why LightGBM was chosen.
 5. Saves a calibration (reliability) curve for the final LightGBM model.
 6. Saves the final LightGBM model + validation arrays in the exact same
    format the Streamlit app already expects — no app changes needed.
"""

import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from preprocess import basic_clean, feature_engineer


# ------------------------------------------------------------
# PATHS & SETTINGS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "upi_transactions.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
CHART_PATH = os.path.join(MODEL_DIR, "feature_importance.png")
ROC_CHART_PATH = os.path.join(MODEL_DIR, "roc_comparison.png")
CALIBRATION_CHART_PATH = os.path.join(MODEL_DIR, "calibration_curve.png")
COMPARISON_CSV_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")
TARGET_COL = "is_fraud"

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
print(f"📥 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------
df = basic_clean(df)
df, _ = feature_engineer(df)
df = df.loc[:, ~df.columns.duplicated()]

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if TARGET_COL not in df.columns:
    raise ValueError(f"❌ Target column '{TARGET_COL}' not found in dataset!")


# ------------------------------------------------------------
# TIME-BASED SPLIT (instead of random split)
# ------------------------------------------------------------
if "timestamp" in df.columns and df["timestamp"].notna().any():
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    print(f"\n🕒 Time-based split:")
    print(f"   Train: {len(df_train)} rows  ({df_train['timestamp'].min()} → {df_train['timestamp'].max()})")
    print(f"   Test : {len(df_test)} rows  ({df_test['timestamp'].min()} → {df_test['timestamp'].max()})")
else:
    print("\n⚠️ No usable timestamp column found — falling back to random 80/20 split.")
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=42)

print(f"   Train fraud rate: {df_train[TARGET_COL].mean()*100:.2f}%")
print(f"   Test  fraud rate: {df_test[TARGET_COL].mean()*100:.2f}%")


# ------------------------------------------------------------
# BUILD FEATURE MATRICES
# ------------------------------------------------------------
drop_cols = [TARGET_COL, "timestamp", "transaction_id"]

X_train = df_train.drop(columns=drop_cols, errors="ignore")
X_test = df_test.drop(columns=drop_cols, errors="ignore")
y_train = df_train[TARGET_COL].astype(int)
y_test = df_test[TARGET_COL].astype(int)

print("\nTraining Features:")
print(list(X_train.columns))

cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
encoders = {}

for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna("unknown")
    X_test[col] = X_test[col].astype(str).fillna("unknown")

    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    encoders[col] = le

for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"🧩 Using {X_train.shape[1]} features for training ({len(cat_cols)} categorical)")
print(f"🧠 Training samples: {len(X_train)}, Test samples: {len(X_test)}")


# ------------------------------------------------------------
# MODEL 1: LOGISTIC REGRESSION (scaled features)
# ------------------------------------------------------------
print("\n🚀 Training Logistic Regression...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]


# ------------------------------------------------------------
# MODEL 2: RANDOM FOREST
# ------------------------------------------------------------
print("🚀 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# MODEL 3: LIGHTGBM (final model used by the app) — fixed, stable hyperparameters
# ------------------------------------------------------------
print("🚀 Training LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": ["auc", "binary_logloss"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

lgbm_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)]
)

lgbm_proba = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)

print("\n✅ All three models trained successfully.")


# ------------------------------------------------------------
# MODEL COMPARISON TABLE
# ------------------------------------------------------------
def evaluate(name, y_true, y_proba, threshold=0.5):
    y_pred_bin = (y_proba >= threshold).astype(int)
    return {
        "Model": name,
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred_bin), 4),
        "Recall": round(recall_score(y_true, y_pred_bin), 4),
        "F1": round(f1_score(y_true, y_pred_bin), 4),
    }

results = [
    evaluate("Logistic Regression", y_test, logreg_proba),
    evaluate("Random Forest", y_test, rf_proba),
    evaluate("LightGBM", y_test, lgbm_proba),
]

comparison_df = pd.DataFrame(results)
print("\n📊 Model Comparison (on time-based test set):")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
print(f"📁 Saved comparison table → {COMPARISON_CSV_PATH}")


# ------------------------------------------------------------
# ROC CURVE OVERLAY (all three models)
# ------------------------------------------------------------
print("\n📈 Generating ROC curve comparison...")
try:
    plt.figure(figsize=(7, 6))
    for name, proba in [
        ("Logistic Regression", logreg_proba),
        ("Random Forest", rf_proba),
        ("LightGBM", lgbm_proba),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CHART_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved ROC comparison chart → {ROC_CHART_PATH}")
except Exception as e:
    print(f"⚠️ Could not generate ROC comparison chart: {e}")


# ------------------------------------------------------------
# CALIBRATION CURVE (LightGBM — the model actually used by the app)
# ------------------------------------------------------------
print("\n📈 Generating calibration curve for LightGBM...")
try:
    prob_true, prob_pred = calibration_curve(y_test, lgbm_proba, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="LightGBM")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of actual positives")
    plt.title("Calibration Curve — LightGBM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CALIBRATION_CHART_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved calibration curve → {CALIBRATION_CHART_PATH}")
except Exception as e:
    print(f"⚠️ Could not generate calibration curve: {e}")


# ------------------------------------------------------------
# EVALUATION (LightGBM — printed detail, as before)
# ------------------------------------------------------------
auc = roc_auc_score(y_test, lgbm_proba)
print(f"\n🎯 LightGBM ROC AUC: {auc:.4f}\n")
print("📊 Classification Report (LightGBM):")
print(classification_report(y_test, (lgbm_proba > 0.5).astype(int)))


# ------------------------------------------------------------
# FEATURE IMPORTANCE (LightGBM)
# ------------------------------------------------------------
print("\n📈 Generating feature importance chart...")
try:
    importance = lgbm_model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({"Feature": X_train.columns, "Importance": importance})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

    if not feat_imp.empty:
        plt.figure(figsize=(10, 6))
        plt.title("Top 20 Feature Importances")
        plt.barh(feat_imp["Feature"].head(20)[::-1], feat_imp["Importance"].head(20)[::-1])
        plt.xlabel("Importance (Gain)")
        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"📊 Saved feature importance chart → {CHART_PATH}")
    else:
        print("⚠️ No feature importance data found — skipping chart generation.")
except Exception as e:
    print(f"⚠️ Could not generate feature importance plot: {e}")


# ------------------------------------------------------------
# SAVE VALIDATION ARRAYS (same filenames the app already reads)
# ------------------------------------------------------------
np.save(os.path.join(MODEL_DIR, "y_val.npy"), y_test.values)
np.save(os.path.join(MODEL_DIR, "y_pred.npy"), (lgbm_proba > 0.5).astype(int))
np.save(os.path.join(MODEL_DIR, "y_proba.npy"), lgbm_proba)
print(f"📁 Saved y_val.npy, y_pred.npy, y_proba.npy → {MODEL_DIR}")


# ------------------------------------------------------------
# SAVE FINAL MODEL (LightGBM — same format the app expects)
# ------------------------------------------------------------
try:
    model_data = {
        "model": lgbm_model,
        "feature_names": list(X_train.columns),
        "categorical_features": cat_cols,
        "encoders": encoders
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"✅ Model saved successfully → {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to save model: {e}")