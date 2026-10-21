import io
import pandas as pd
from app.config import DEMO_TRANSACTIONS_PATH

def test_predict_known_samples(client):
    # Load demo transactions to extract a known fraud and known genuine sample
    demo_df = pd.read_csv(DEMO_TRANSACTIONS_PATH)
    fraud_row = demo_df[demo_df["Class"] == 1].iloc[0]
    genuine_row = demo_df[demo_df["Class"] == 0].iloc[0]

    feature_cols = [c for c in demo_df.columns if c != "Class"]

    # 1. Predict known fraud
    fraud_payload = {"features": {col: float(fraud_row[col]) for col in feature_cols}}
    res_fraud = client.post("/api/predict", json=fraud_payload)
    assert res_fraud.status_code == 200
    data_fraud = res_fraud.json()
    assert data_fraud["probability"] >= 0.5
    assert data_fraud["is_fraud"] is True
    assert data_fraud["risk_level"] == "high"
    assert len(data_fraud["top_signals"]) == 5
    for sig in data_fraud["top_signals"]:
        assert "feature" in sig
        assert "importance" in sig
        assert "deviation" in sig
        assert "score" in sig
        assert "description" in sig

    # 2. Predict known genuine
    genuine_payload = {"features": {col: float(genuine_row[col]) for col in feature_cols}}
    res_genuine = client.post("/api/predict", json=genuine_payload)
    assert res_genuine.status_code == 200
    data_genuine = res_genuine.json()
    assert data_genuine["probability"] < 0.2
    assert data_genuine["is_fraud"] is False
    assert data_genuine["risk_level"] == "low"

def test_predict_validation_errors(client):
    # Missing required features
    incomplete_payload = {"features": {"Time": 100.0, "Amount": 50.0}}
    res = client.post("/api/predict", json=incomplete_payload)
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data

def test_batch_prediction(client):
    with open(DEMO_TRANSACTIONS_PATH, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("demo_transactions.csv", file_bytes, "text/csv")}
    res = client.post("/api/predict-batch", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows"] == 100
    assert data["fraud_count"] > 0
    assert data["genuine_count"] > 0
    assert len(data["rows"]) == 100

    # Since demo_transactions has a Class column, evaluation metrics must be populated
    eval_m = data["evaluation_metrics"]
    assert eval_m is not None
    assert eval_m["fraud_actual"] == 25
    assert eval_m["genuine_actual"] == 75
    assert eval_m["tp"] > 0
    assert eval_m["precision"] > 0.5
    assert eval_m["recall"] > 0.5

def test_batch_prediction_invalid_file(client):
    # Non-csv upload
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    res = client.post("/api/predict-batch", files=files)
    assert res.status_code == 400
