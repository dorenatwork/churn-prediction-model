"""FastAPI service exposing the trained churn model. Run with: uvicorn src.serve:app --reload"""
import json
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import (
    CATEGORICAL_FEATURES,
    METADATA_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_ARTIFACT_PATH,
)
from src.schemas import ChurnRequest, ChurnResponse, HealthResponse

logger = logging.getLogger("churn_api")

_model = None
_preprocessor = None
_metadata: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="E-commerce Churn Prediction API",
    description="Predicts the probability that a customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)

# Business risk bands used to turn a probability into an actionable label.
LOW_RISK_MAX = 0.33
MEDIUM_RISK_MAX = 0.66


def _risk_label(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "low_risk"
    if probability < MEDIUM_RISK_MAX:
        return "medium_risk"
    return "high_risk"


def _load_artifacts() -> None:
    global _model, _preprocessor, _metadata
    if not MODEL_ARTIFACT_PATH.exists() or not PREPROCESSOR_ARTIFACT_PATH.exists():
        logger.warning("Model artifacts not found at %s — train a model first.", MODEL_ARTIFACT_PATH)
        return
    _model = joblib.load(MODEL_ARTIFACT_PATH)
    _preprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
    if METADATA_ARTIFACT_PATH.exists():
        _metadata = json.loads(METADATA_ARTIFACT_PATH.read_text())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _model is None:
        return HealthResponse(status="model_not_loaded")
    version = _metadata.get("model_version")
    return HealthResponse(
        status="ok",
        model_name=_metadata.get("model_name"),
        model_version=str(version) if version is not None else None,
    )


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest) -> ChurnResponse:
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model first (see README).")

    row = request.model_dump()
    frame = pd.DataFrame([row])

    for col in NUMERIC_FEATURES:
        if col not in frame.columns:
            frame[col] = None
    for col in CATEGORICAL_FEATURES:
        if col not in frame.columns:
            frame[col] = None

    X = _preprocessor.transform(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    probability = float(_model.predict_proba(X)[0, 1])

    return ChurnResponse(churn_probability=round(probability, 4), prediction=_risk_label(probability))
