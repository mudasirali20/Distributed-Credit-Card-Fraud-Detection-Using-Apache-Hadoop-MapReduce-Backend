import os
import io
import math
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from app.config import (
    MODEL_PATH,
    THRESHOLD_DEFAULT,
    RISK_LOW_CEILING,
    MAX_BATCH_ROWS,
    MAX_BATCH_FILE_BYTES,
)
from app.schemas import (
    PredictResponse,
    SignalExplanation,
    BatchPredictResponse,
    BatchRowResult,
    BatchMetrics,
    REQUIRED_FEATURES,
)

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "V14": "Primary PCA factor capturing high-magnitude anomaly deviations",
    "V12": "Key latent dimension linked to atypical authorization patterns",
    "V10": "Secondary PCA indicator of transactional distance from centroid",
    "V17": "Discriminative subspace component sensitive to fraudulent sequences",
    "V4": "Positively correlated component indicating elevated transaction velocity",
    "V11": "Structural divergence signal in account behavioral profile",
    "V16": "Transaction latency and authentication consistency factor",
    "V7": "Spending consistency score across merchant category clusters",
    "V3": "Baseline transaction integrity component",
    "Amount": "Transaction monetary amount deviation from baseline expenditure",
    "Time": "Elapsed seconds since benchmark reference timestamp",
}

class FraudModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features: List[str] = []
        self.threshold: float = THRESHOLD_DEFAULT
        self.genuine_medians: Dict[str, float] = {}
        self.genuine_stds: Dict[str, float] = {}
        self.feature_importances: List[Dict[str, Any]] = []
        self._loaded: bool = False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Ensure fraud_model.joblib is generated or placed in the data/ folder."
            )

        data = joblib.load(MODEL_PATH)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid model structure in {MODEL_PATH}. Expected a dictionary.")

        self.model = data.get("model")
        self.scaler = data.get("scaler")
        self.features = data.get("features", REQUIRED_FEATURES)
        self.threshold = float(data.get("threshold", THRESHOLD_DEFAULT))
        self.genuine_medians = data.get("genuine_medians", {})
        self.genuine_stds = data.get("genuine_stds", {})

        if self.model is None or self.scaler is None:
            raise ValueError("Model dictionary must contain 'model' and 'scaler' keys.")

        # Compute and sort feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feat_imp = [
                {"name": name, "importance": float(imp)}
                for name, imp in zip(self.features, importances)
            ]
            feat_imp.sort(key=lambda x: x["importance"], reverse=True)
            self.feature_importances = feat_imp

        self._loaded = True

    def get_top_feature_importances(self, top_n: int = 10) -> List[Dict[str, Any]]:
        if not self._loaded:
            self.load()
        return self.feature_importances[:top_n]

    def _determine_risk_level(self, probability: float) -> str:
        if probability < RISK_LOW_CEILING:
            return "low"
        elif probability < self.threshold:
            return "medium"
        return "high"

    def predict_single(self, features: Dict[str, float]) -> PredictResponse:
        if not self._loaded:
            self.load()

        # Build feature vector in exact training order
        row_values = [float(features[col]) for col in self.features]
        df_row = pd.DataFrame([row_values], columns=self.features)

        # Scale Time and Amount ONLY
        scaled_time_amt = self.scaler.transform(df_row[["Time", "Amount"]])
        df_row_scaled = df_row.copy()
        df_row_scaled[["Time", "Amount"]] = scaled_time_amt

        # Predict probability
        proba = float(self.model.predict_proba(df_row_scaled)[0, 1])
        is_fraud = proba >= self.threshold
        risk = self._determine_risk_level(proba)

        # Calculate top 5 signals explaining the score
        # feature_importance * |standardised deviation from genuine-transaction median|
        signals: List[SignalExplanation] = []
        importances_map = {item["name"]: item["importance"] for item in self.feature_importances}

        for col in self.features:
            raw_val = float(features[col])
            med = self.genuine_medians.get(col, 0.0)
            std = self.genuine_stds.get(col, 1.0)
            if std <= 1e-7:
                std = 1.0

            std_dev = (raw_val - med) / std
            imp = importances_map.get(col, 0.01)
            score = imp * abs(std_dev)

            desc = FEATURE_DESCRIPTIONS.get(
                col, f"Standardised deviation of {std_dev:+.2f} standard errors from genuine baseline"
            )

            signals.append(
                SignalExplanation(
                    feature=col,
                    importance=round(imp, 5),
                    deviation=round(std_dev, 4),
                    score=round(score, 5),
                    description=desc,
                )
            )

        signals.sort(key=lambda s: s.score, reverse=True)
        top_5 = signals[:5]

        return PredictResponse(
            probability=round(proba, 5),
            is_fraud=is_fraud,
            risk_level=risk,
            threshold=self.threshold,
            top_signals=top_5,
        )

    def predict_batch(self, csv_bytes: bytes) -> BatchPredictResponse:
        if not self._loaded:
            self.load()

        if len(csv_bytes) > MAX_BATCH_FILE_BYTES:
            raise ValueError(f"File exceeds maximum size of {MAX_BATCH_FILE_BYTES // (1024*1024)} MB.")

        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {str(e)}")

        if len(df) > MAX_BATCH_ROWS:
            raise ValueError(f"Batch contains {len(df)} rows. Maximum allowed is {MAX_BATCH_ROWS} rows.")

        missing_cols = [c for c in self.features if c not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {', '.join(missing_cols[:5])}...")

        has_actual_class = "Class" in df.columns
        actual_classes = df["Class"].astype(int).tolist() if has_actual_class else None

        # Build feature matrix in training order
        X = df[self.features].copy()
        X[["Time", "Amount"]] = self.scaler.transform(X[["Time", "Amount"]])

        probs = self.model.predict_proba(X)[:, 1]

        rows: List[BatchRowResult] = []
        fraud_count = 0

        for i in range(len(df)):
            p = float(probs[i])
            is_f = p >= self.threshold
            if is_f:
                fraud_count += 1
            risk = self._determine_risk_level(p)
            actual = int(actual_classes[i]) if actual_classes is not None else None

            rows.append(
                BatchRowResult(
                    row_id=i + 1,
                    Time=float(df.iloc[i]["Time"]),
                    Amount=float(df.iloc[i]["Amount"]),
                    probability=round(p, 5),
                    is_fraud=is_f,
                    risk_level=risk,
                    actual_class=actual,
                )
            )

        total_rows = len(df)
        genuine_count = total_rows - fraud_count
        fraud_rate = (fraud_count / total_rows * 100.0) if total_rows > 0 else 0.0

        eval_metrics = None
        if has_actual_class and actual_classes is not None:
            y_true = np.array(actual_classes)
            y_pred = (probs >= self.threshold).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            eval_metrics = BatchMetrics(
                total=total_rows,
                fraud_actual=int(np.sum(y_true == 1)),
                genuine_actual=int(np.sum(y_true == 0)),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
            )

        return BatchPredictResponse(
            total_rows=total_rows,
            fraud_count=fraud_count,
            genuine_count=genuine_count,
            fraud_rate_percent=round(fraud_rate, 4),
            evaluation_metrics=eval_metrics,
            rows=rows,
        )

model_service = FraudModelService()
