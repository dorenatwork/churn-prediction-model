"""Smoke tests for the training building blocks — skips the full RandomizedSearchCV sweep (too slow for a unit test)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.metrics import compute_metrics
from src.train import build_preprocessor


def _synthetic_frame(n=100, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 100, n) for col in NUMERIC_FEATURES}
    # inject missingness like the real dataset
    for col in NUMERIC_FEATURES:
        mask = rng.random(n) < 0.05
        data[col] = np.where(mask, np.nan, data[col])
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame(data)
    y = rng.integers(0, 2, n)
    return df, y


def test_preprocessor_handles_missing_values():
    df, _ = _synthetic_frame()
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    assert not np.isnan(X).any()
    assert X.shape[0] == len(df)


def test_pipeline_fits_and_predicts_probabilities():
    df, y = _synthetic_frame()
    preprocessor = build_preprocessor()
    pipeline = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=200))])
    pipeline.fit(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y)
    prob = pipeline.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    assert prob.shape == (len(df),)
    assert ((prob >= 0) & (prob <= 1)).all()
    metrics = compute_metrics(y, prob)
    assert 0 <= metrics["roc_auc"] <= 1
