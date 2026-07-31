"""Train LR/RF/XGBoost candidates, track in MLflow, select best by val PR-AUC, register, score test once. Usage: python -m src.train"""
from __future__ import annotations

import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    METADATA_ARTIFACT_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_ARTIFACT_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_ARTIFACT_PATH,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    TARGET_COL,
)
from src.metrics import best_threshold_by_f1, compute_metrics, save_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]
    return X_train, y_train, X_val, y_val, X_test, y_test


# Small search space per model — ~4k rows on CPU doesn't need a wide sweep
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [4, 8, 12, None],
            "clf__min_samples_leaf": [1, 2, 5],
        },
    ),
    "xgboost": (
        XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
            device="cpu",
        ),
        {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [3, 4, 6, 8],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.85, 1.0],
        },
    ),
}


def _scale_pos_weight(y_train: pd.Series) -> float:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    return neg / pos


def train_candidate(name, estimator, param_dist, X_train, y_train, X_val, y_val):
    preprocessor = build_preprocessor()

    if name == "xgboost":
        estimator.set_params(scale_pos_weight=_scale_pos_weight(y_train))

    pipeline = Pipeline([("prep", preprocessor), ("clf", estimator)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=15,
        scoring="average_precision",  # PR-AUC
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    val_prob = search.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val.values, val_prob, threshold=0.5)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({f"cv_best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_param("model_type", name)
        mlflow.log_metric("cv_best_pr_auc", search.best_score_)
        for k, v in val_metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    mlflow.log_metric(f"val_{k}_{sub_k}", sub_v)
            else:
                mlflow.log_metric(f"val_{k}", v)
        mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model", serialization_format="pickle")
        run_id = mlflow.active_run().info.run_id

    logger.info("%s: val PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f", name, val_metrics["pr_auc"], val_metrics["roc_auc"], val_metrics["f1"])
    return {
        "name": name,
        "pipeline": search.best_estimator_,
        "val_metrics": val_metrics,
        "run_id": run_id,
        "best_params": search.best_params_,
    }


def run() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    results = []
    for name, (estimator, param_dist) in CANDIDATES.items():
        result = train_candidate(name, estimator, param_dist, X_train, y_train, X_val, y_val)
        results.append(result)

    best = max(results, key=lambda r: r["val_metrics"]["pr_auc"])
    logger.info("Selected model: %s (val PR-AUC=%.4f)", best["name"], best["val_metrics"]["pr_auc"])

    threshold = best_threshold_by_f1(y_val.values, best["pipeline"].predict_proba(X_val)[:, 1])
    logger.info("Chosen decision threshold (from val, F1-optimal): %.2f", threshold)

    # Test set scored exactly once, after model + threshold were fixed on train/val
    test_prob = best["pipeline"].predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test.values, test_prob, threshold=threshold)
    logger.info("Held-out TEST metrics: %s", json.dumps(test_metrics, indent=2))

    with mlflow.start_run(run_name=f"{best['name']}_final_test_eval"):
        mlflow.log_param("model_type", best["name"])
        mlflow.log_param("selected_from_run_id", best["run_id"])
        mlflow.log_param("decision_threshold", threshold)
        for k, v in test_metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    mlflow.log_metric(f"test_{k}_{sub_k}", sub_v)
            else:
                mlflow.log_metric(f"test_{k}", v)
        model_info = mlflow.sklearn.log_model(
            best["pipeline"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )

    # Plain joblib copy so the API doesn't need a live MLflow server
    full_pipeline = best["pipeline"]
    preprocessor = full_pipeline.named_steps["prep"]
    classifier = full_pipeline.named_steps["clf"]

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_ARTIFACT_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_ARTIFACT_PATH)

    metadata = {
        "model_name": best["name"],
        "model_version": model_info.registered_model_version if hasattr(model_info, "registered_model_version") else None,
        "decision_threshold": threshold,
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "best_params": best["best_params"],
        "mlflow_run_id": best["run_id"],
    }
    METADATA_ARTIFACT_PATH.write_text(json.dumps(metadata, indent=2))

    comparison = {r["name"]: r["val_metrics"] for r in results}
    save_metrics(comparison, REPORTS_DIR / "model_comparison.json")
    save_metrics(test_metrics, REPORTS_DIR / "test_metrics.json")

    logger.info("Saved model artifacts to %s and metadata to %s", MODEL_ARTIFACT_PATH, METADATA_ARTIFACT_PATH)


if __name__ == "__main__":
    run()
