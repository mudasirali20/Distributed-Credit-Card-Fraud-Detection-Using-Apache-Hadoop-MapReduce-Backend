import os
import sys
import shutil
import json
import urllib.request
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Base Directory: {BASE_DIR}")
print(f"Data Directory: {DATA_DIR}")

# 1. Copy pre-existing files into data/
files_to_copy = [
    "summary.json",
    "hadoop_by_class.csv",
    "hadoop_by_amount.csv",
    "hadoop_by_hour.csv",
    "model_metrics.csv",
    "top_suspicious.csv",
    "dashboard.png",
]

for filename in files_to_copy:
    src = os.path.join(BASE_DIR, filename)
    dst = os.path.join(DATA_DIR, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {filename} to data/")
    elif os.path.exists(dst):
        print(f"Found existing {filename} in data/")
    else:
        print(f"Warning: {filename} not found in root or data/")

# 2. Check if model and demo transactions exist
model_path = os.path.join(DATA_DIR, "fraud_model.joblib")
demo_path = os.path.join(DATA_DIR, "demo_transactions.csv")

if os.path.exists(model_path) and os.path.exists(demo_path):
    print("fraud_model.joblib and demo_transactions.csv already exist. Skipping training.")
    sys.exit(0)

# 3. Check for creditcard.csv or download it
csv_path = os.path.join(BASE_DIR, "creditcard.csv")
if not os.path.exists(csv_path):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    print(f"Downloading creditcard.csv from {url}...")
    def report_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent % 20 == 0:
            print(f"Download progress: {percent}%", end="\r")
    urllib.request.urlretrieve(url, csv_path, reporthook=report_hook)
    print("\nDownload complete.")

print("Loading creditcard.csv...")
df = pd.read_csv(csv_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. Frauds: {df['Class'].sum()}")

# 4. Stratified Split matching Colab notebook: random_state=42, test_size=0.2
X = df.drop(columns=["Class"])
y = df["Class"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fit scaler on Time and Amount only
scaler = StandardScaler().fit(X_train[["Time", "Amount"]])

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[["Time", "Amount"]] = scaler.transform(X_train[["Time", "Amount"]])
X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

# Compute genuine statistics for explainability (median and std of genuine transactions)
genuine_mask = (y_train == 0)
genuine_data = X_train[genuine_mask]
genuine_medians = {col: float(genuine_data[col].median()) for col in feature_names}
genuine_stds = {col: float(max(genuine_data[col].std(), 1e-6)) for col in feature_names}

print("Training Random Forest model (matching Colab parameters)...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train_scaled, y_train)

# Evaluate on test set
p_test = rf.predict_proba(X_test_scaled)[:, 1]
y_pred = (p_test >= 0.5).astype(int)

prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, p_test)
pr_auc = average_precision_score(y_test, p_test)

print(f"Trained Random Forest Evaluation:")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"  PR-AUC:    {pr_auc:.4f}")

# 5. Save model package
model_payload = {
    "model": rf,
    "scaler": scaler,
    "features": feature_names,
    "threshold": 0.5,
    "genuine_medians": genuine_medians,
    "genuine_stds": genuine_stds,
}
joblib.dump(model_payload, model_path)
print(f"Saved model package to {model_path}")

# 6. Generate 100 demo transactions (unscaled, 25 fraud and 75 genuine)
# Combine X_test and y_test
test_df = X_test.copy()
test_df["Class"] = y_test

fraud_samples = test_df[test_df["Class"] == 1].sample(n=25, random_state=42)
genuine_samples = test_df[test_df["Class"] == 0].sample(n=75, random_state=42)
demo_df = pd.concat([fraud_samples, genuine_samples]).sample(frac=1.0, random_state=42).reset_index(drop=True)
demo_df.to_csv(demo_path, index=False)
print(f"Saved 100 demo transactions to {demo_path} ({len(demo_df[demo_df['Class'] == 1])} fraud, {len(demo_df[demo_df['Class'] == 0])} genuine)")

print("Data preparation completed successfully!")
