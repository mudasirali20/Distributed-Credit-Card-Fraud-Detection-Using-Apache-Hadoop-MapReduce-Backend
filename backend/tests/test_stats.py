def test_summary(client):
    response = client.get("/api/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 284807
    assert data["fraud_transactions"] == 492
    assert abs(data["fraud_rate_percent"] - 0.1727) < 0.001
    assert data["hadoop_matches_pandas"] is True
    assert "rule_detector" in data
    assert data["rule_detector"]["TP"] == 403
    assert data["rule_detector"]["FP"] == 652

def test_hadoop_class_stats(client):
    response = client.get("/api/stats/class")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) >= 2
    fraud_row = next((r for r in rows if r["key"] == 1 or r["key"] == "1"), None)
    genuine_row = next((r for r in rows if r["key"] == 0 or r["key"] == "0"), None)
    assert fraud_row is not None
    assert genuine_row is not None
    assert fraud_row["count"] == 492
    assert genuine_row["count"] == 284315

def test_hadoop_amount_stats(client):
    response = client.get("/api/stats/amount")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 6
    assert any("0-10" in r["bucket"] for r in rows)

def test_hadoop_hour_stats(client):
    response = client.get("/api/stats/hour")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 24

def test_metrics(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    rows = response.json()
    model_names = [r["model"] for r in rows]
    assert any("Random Forest" in m for m in model_names)
    assert any("Logistic Regression" in m for m in model_names)
    assert any("Rule Detector" in m for m in model_names)

def test_suspicious(client):
    response = client.get("/api/suspicious")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) > 0
    assert "fraud_probability" in rows[0]

def test_samples(client):
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) == 100
    assert "id" in samples[0]
    assert "Class" in samples[0]

    sample_id = samples[0]["id"]
    single_res = client.get(f"/api/samples/{sample_id}")
    assert single_res.status_code == 200
    assert single_res.json()["id"] == sample_id

def test_feature_importance(client):
    response = client.get("/api/feature-importance")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 10
    # Assert sorted descending
    for i in range(len(items) - 1):
        assert items[i]["importance"] >= items[i + 1]["importance"]
