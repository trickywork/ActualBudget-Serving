from fastapi.testclient import TestClient

from app.main import app
from app.schemas import PredictResponse


class DummyBackendOutput:
    labels = ["Food & Dining"]
    probabilities = [[0.8, 0.1, 0.1]]
    classes = ["Food & Dining", "Shopping & Retail", "Entertainment & Recreation"]


class DummyBackend:
    def predict(self, frame):
        return DummyBackendOutput()

    def providers(self):
        return ["dummy"]


def test_predict_endpoint_contract(monkeypatch):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
            "country": "US",
            "currency": "USD",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_category_id"] == "Food & Dining"
    assert len(payload["top_categories"]) == 3


def test_predict_batch_endpoint_contract(monkeypatch):
    monkeypatch.setattr("app.main.get_backend", lambda: DummyBackend())
    client = TestClient(app)
    response = client.post(
        "/predict_batch",
        json={
            "items": [
                {
                    "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
                    "country": "US",
                    "currency": "USD",
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "items" in payload
    assert payload["items"][0]["predicted_category_id"] == "Food & Dining"
