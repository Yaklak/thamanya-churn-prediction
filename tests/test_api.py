import pytest
from starlette.testclient import TestClient


class DummyModel:
    def predict(self, X):
        # Return zeros for deterministic tests
        return [0] * len(X)

    def predict_proba(self, X):
        # Return balanced probabilities
        return [[0.5, 0.5]] * len(X)


@pytest.fixture()
def client(monkeypatch):
    import api.app as api_module

    # Backup originals
    original_model = getattr(api_module, "model", None)
    original_schema = getattr(api_module, "input_schema", None)
    original_load = getattr(api_module, "_load", None)

    # Prevent real startup loader from overwriting our patches
    if hasattr(api_module, "_load"):
        monkeypatch.setattr(api_module, "_load", lambda: None)

    # Build client (startup runs but does nothing)
    test_client = TestClient(api_module.app)

    # Now patch globals for tests
    api_module.model = DummyModel()
    api_module.input_schema = ["a", "b", "c"]

    yield test_client

    # Restore originals
    if original_load is not None:
        monkeypatch.setattr(api_module, "_load", original_load)
    api_module.model = original_model
    api_module.input_schema = original_schema


def test_health(client: TestClient):
    """Basic health endpoint should respond with status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"


def test_model_schema(client: TestClient):
    """Test the /model/schema endpoint returns our patched schema."""
    response = client.get("/model/schema")
    assert response.status_code == 200
    body = response.json()
    assert body.get("expected_columns") == ["a", "b", "c"]
    assert body.get("count") == 3


def test_predict_missing_columns(client: TestClient):
    """Predict should accept payloads even if columns are missing (API auto-aligns)."""
    payload = {"a": 1, "b": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body


def test_predict_extra_columns(client: TestClient):
    """Predict should accept payloads with extra columns (API ignores extras)."""
    payload = {"a": 1, "b": 2, "c": 3, "d": 4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
