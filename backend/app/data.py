import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from app.config import (
    DATA_DIR,
    SUMMARY_PATH,
    HADOOP_CLASS_PATH,
    HADOOP_AMOUNT_PATH,
    HADOOP_HOUR_PATH,
    MODEL_METRICS_PATH,
    TOP_SUSPICIOUS_PATH,
    DEMO_TRANSACTIONS_PATH,
)

class DataStore:
    def __init__(self):
        self._summary: Optional[Dict[str, Any]] = None
        self._hadoop_class: Optional[List[Dict[str, Any]]] = None
        self._hadoop_amount: Optional[List[Dict[str, Any]]] = None
        self._hadoop_hour: Optional[List[Dict[str, Any]]] = None
        self._model_metrics: Optional[List[Dict[str, Any]]] = None
        self._top_suspicious: Optional[List[Dict[str, Any]]] = None
        self._demo_transactions: Optional[List[Dict[str, Any]]] = None

    def verify_files(self) -> None:
        required = [
            ("summary.json", SUMMARY_PATH),
            ("hadoop_by_class.csv", HADOOP_CLASS_PATH),
            ("hadoop_by_amount.csv", HADOOP_AMOUNT_PATH),
            ("hadoop_by_hour.csv", HADOOP_HOUR_PATH),
            ("model_metrics.csv", MODEL_METRICS_PATH),
            ("top_suspicious.csv", TOP_SUSPICIOUS_PATH),
            ("demo_transactions.csv", DEMO_TRANSACTIONS_PATH),
        ]
        missing = [name for name, path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required data file(s) in {DATA_DIR}: {', '.join(missing)}. "
                "Ensure files from Colab are placed in the data/ folder."
            )

    def load_all(self) -> None:
        self.verify_files()

        # Load summary.json
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            self._summary = json.load(f)

        # Load Hadoop stats
        self._hadoop_class = pd.read_csv(HADOOP_CLASS_PATH).to_dict(orient="records")
        self._hadoop_amount = pd.read_csv(HADOOP_AMOUNT_PATH).to_dict(orient="records")
        self._hadoop_hour = pd.read_csv(HADOOP_HOUR_PATH).to_dict(orient="records")

        # Load model metrics and append rule detector metrics from summary.json
        metrics_df = pd.read_csv(MODEL_METRICS_PATH)
        metrics_records = metrics_df.to_dict(orient="records")
        if self._summary and "rule_detector" in self._summary:
            rd = self._summary["rule_detector"]
            rule_row = {
                "model": "Hadoop Rule Detector (V14<-4 or V12<-4)",
                "precision": rd.get("precision", 0.0),
                "recall": rd.get("recall", 0.0),
                "f1": rd.get("f1", 0.0),
                "roc_auc": None,
                "pr_auc": None,
                "train_seconds": None,
            }
            metrics_records.append(rule_row)
        self._model_metrics = metrics_records

        # Load top suspicious
        self._top_suspicious = pd.read_csv(TOP_SUSPICIOUS_PATH).to_dict(orient="records")

        # Load demo transactions with 1-based id
        demo_df = pd.read_csv(DEMO_TRANSACTIONS_PATH)
        demo_records = []
        for idx, row in demo_df.iterrows():
            rec = row.to_dict()
            rec["id"] = int(idx + 1)
            rec["Class"] = int(rec.get("Class", 0))
            demo_records.append(rec)
        self._demo_transactions = demo_records

    def get_summary(self) -> Dict[str, Any]:
        if self._summary is None:
            self.load_all()
        return self._summary

    def get_hadoop_class_stats(self) -> List[Dict[str, Any]]:
        if self._hadoop_class is None:
            self.load_all()
        return self._hadoop_class

    def get_hadoop_amount_stats(self) -> List[Dict[str, Any]]:
        if self._hadoop_amount is None:
            self.load_all()
        return self._hadoop_amount

    def get_hadoop_hour_stats(self) -> List[Dict[str, Any]]:
        if self._hadoop_hour is None:
            self.load_all()
        return self._hadoop_hour

    def get_metrics(self) -> List[Dict[str, Any]]:
        if self._model_metrics is None:
            self.load_all()
        return self._model_metrics

    def get_suspicious(self) -> List[Dict[str, Any]]:
        if self._top_suspicious is None:
            self.load_all()
        return self._top_suspicious

    def get_samples(self) -> List[Dict[str, Any]]:
        if self._demo_transactions is None:
            self.load_all()
        return self._demo_transactions

    def get_sample_by_id(self, sample_id: int) -> Optional[Dict[str, Any]]:
        samples = self.get_samples()
        for s in samples:
            if s.get("id") == sample_id:
                return s
        return None

data_store = DataStore()
