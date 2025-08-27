from fastapi.testclient import TestClient


def test_root(client: TestClient):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "docs": "/docs", "health": "/health"}


def test_health(client: TestClient):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_model_info(client: TestClient):
    """Test the model_info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["loaded"] is True
    assert response.json()["model_class"] == "LogisticRegression"


def test_model_schema(client: TestClient):
    """Test the model_schema endpoint."""
    response = client.get("/model/schema")
    assert response.status_code == 200
    assert response.json()["expected_columns"] == ["a", "b", "c"]


def test_predict(client: TestClient):
    """Test the predict endpoint with a valid payload."""
    payload = {"a": 1, "b": 2, "c": 3}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_probability" in response.json()


def test_predict_missing_columns(client: TestClient):
    """Test the predict endpoint with missing columns."""
    payload = {"a": 1, "b": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_extra_columns(client: TestClient):
    """Test the predict endpoint with extra columns."""
    payload = {"a": 1, "b": 2, "c": 3, "d": 4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_no_model(client: TestClient):
    """Test the predict endpoint when no model is loaded."""
    # To simulate no model, we can point to a non-existent path
    from api import app as api_module

    original_model_path = api_module.MODEL_PATH
    api_module.MODEL_PATH = api_module.BEST_DIR / "non_existent_model.joblib"
    api_module._load()

    payload = {"a": 1, "b": 2, "c": 3}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503

    # Restore the original model path for other tests
    api_module.MODEL_PATH = original_model_path
    api_module._load()
