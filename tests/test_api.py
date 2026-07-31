"""API tests. Uses a tiny in-memory model/preprocessor so the suite doesn't depend on a prior training run."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import serve
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture()
def client(tmp_path, monkeypatch):
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, NUMERIC_FEATURES), ("cat", categorical_pipe, CATEGORICAL_FEATURES)]
    )

    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(0, 48, n),
            "orders": rng.integers(0, 30, n),
            "average_order_value": rng.uniform(10, 200, n),
            "days_since_last_order": rng.integers(0, 200, n),
            "complain": rng.integers(0, 2, n),
            "satisfaction_score": rng.integers(1, 6, n),
            "hour_spend_on_app": rng.uniform(0, 5, n),
            "number_of_device_registered": rng.integers(1, 6, n),
            "number_of_address": rng.integers(1, 10, n),
            "warehouse_to_home": rng.uniform(5, 40, n),
            "order_amount_hike_pct": rng.uniform(10, 25, n),
            "coupon_used": rng.integers(0, 5, n),
            "subscription_type": rng.choice(["Mobile", "Fashion"], n),
            "country": rng.choice(["Tier 1", "Tier 2", "Tier 3"], n),
            "gender": rng.choice(["Male", "Female"], n),
            "marital_status": rng.choice(["Single", "Married"], n),
            "preferred_login_device": rng.choice(["Mobile Phone", "Computer"], n),
            "preferred_payment_mode": rng.choice(["Credit Card", "UPI"], n),
        }
    )
    y = rng.integers(0, 2, n)

    X = preprocessor.fit_transform(df)
    model = LogisticRegression().fit(X, y)

    model_path = tmp_path / "model.joblib"
    preproc_path = tmp_path / "preprocessor.joblib"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preproc_path)

    monkeypatch.setattr(serve, "MODEL_ARTIFACT_PATH", model_path, raising=False)
    monkeypatch.setattr(serve, "PREPROCESSOR_ARTIFACT_PATH", preproc_path, raising=False)
    monkeypatch.setattr(serve, "_model", None)
    monkeypatch.setattr(serve, "_preprocessor", None)

    def _fake_load():
        serve._model = joblib.load(model_path)
        serve._preprocessor = joblib.load(preproc_path)

    monkeypatch.setattr(serve, "_load_artifacts", _fake_load)

    with TestClient(serve.app) as c:
        yield c


def test_health_when_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_payload(client):
    # "complain" replaces the doc's "support_tickets" — see README
    payload = {
        "tenure_months": 14,
        "orders": 8,
        "average_order_value": 72.50,
        "days_since_last_order": 63,
        "complain": 1,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "churn_probability" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in {"low_risk", "medium_risk", "high_risk"}


def test_predict_rejects_negative_values(client):
    payload = {
        "tenure_months": -1,
        "orders": 8,
        "average_order_value": 72.50,
        "days_since_last_order": 63,
        "complain": 1,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_complain_outside_0_1(client):
    payload = {
        "tenure_months": 12,
        "orders": 8,
        "average_order_value": 72.5,
        "days_since_last_order": 10,
        "complain": 9,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_without_loaded_model_returns_503(monkeypatch):
    monkeypatch.setattr(serve, "_model", None)
    monkeypatch.setattr(serve, "_preprocessor", None)
    monkeypatch.setattr(serve, "_load_artifacts", lambda: None)
    with TestClient(serve.app) as c:
        resp = c.post(
            "/predict",
            json={
                "tenure_months": 1,
                "orders": 1,
                "average_order_value": 1,
                "days_since_last_order": 1,
                "complain": 1,
            },
        )
    assert resp.status_code == 503
