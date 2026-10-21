import os
from pathlib import Path

# Locate project base directory
# backend/app/config.py -> backend/app -> backend -> root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

# Fallback: if data is located alongside backend
if not DATA_DIR.exists():
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Model and dataset paths
MODEL_PATH = DATA_DIR / "fraud_model.joblib"
SUMMARY_PATH = DATA_DIR / "summary.json"
HADOOP_CLASS_PATH = DATA_DIR / "hadoop_by_class.csv"
HADOOP_AMOUNT_PATH = DATA_DIR / "hadoop_by_amount.csv"
HADOOP_HOUR_PATH = DATA_DIR / "hadoop_by_hour.csv"
MODEL_METRICS_PATH = DATA_DIR / "model_metrics.csv"
TOP_SUSPICIOUS_PATH = DATA_DIR / "top_suspicious.csv"
DEMO_TRANSACTIONS_PATH = DATA_DIR / "demo_transactions.csv"
DASHBOARD_IMAGE_PATH = DATA_DIR / "dashboard.png"

# Risk classification levels
THRESHOLD_DEFAULT = 0.5
RISK_LOW_CEILING = 0.2

# Limits for batch uploads
MAX_BATCH_ROWS = 5000
MAX_BATCH_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
