from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_file: str

class RuleDetectorMetrics(BaseModel):
    threshold_V14: float
    threshold_V12: float
    TP: int
    FP: int
    FN: int
    TN: int
    precision: float
    recall: float
    f1: float

class SummaryResponse(BaseModel):
    total_transactions: int
    fraud_transactions: int
    fraud_rate_percent: float
    hadoop_matches_pandas: bool
    hadoop_class_job_seconds: float
    pandas_seconds: float
    rule_detector: RuleDetectorMetrics

class ModelMetricRow(BaseModel):
    model: str
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    train_seconds: Optional[float] = None

class FeatureImportanceItem(BaseModel):
    name: str
    importance: float

class SignalExplanation(BaseModel):
    feature: str
    importance: float
    deviation: float
    score: float
    description: str

REQUIRED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

class PredictRequest(BaseModel):
    features: Dict[str, float]

    @model_validator(mode="after")
    def validate_features(self) -> "PredictRequest":
        provided = set(self.features.keys())
        missing = [f for f in REQUIRED_FEATURES if f not in provided]
        if missing:
            raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
        return self

class PredictResponse(BaseModel):
    probability: float
    is_fraud: bool
    risk_level: str
    threshold: float
    top_signals: List[SignalExplanation]

class BatchRowResult(BaseModel):
    row_id: int
    Time: float
    Amount: float
    probability: float
    is_fraud: bool
    risk_level: str
    actual_class: Optional[int] = None

class BatchMetrics(BaseModel):
    total: int
    fraud_actual: int
    genuine_actual: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

class BatchPredictResponse(BaseModel):
    total_rows: int
    fraud_count: int
    genuine_count: int
    fraud_rate_percent: float
    evaluation_metrics: Optional[BatchMetrics] = None
    rows: List[BatchRowResult]
