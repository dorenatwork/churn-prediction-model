"""SHAP explainability + a plain-language summary. Requires a model already trained via python -m src.train."""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    MODEL_ARTIFACT_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_ARTIFACT_PATH,
    REPORTS_DIR,
)


def _feature_names(preprocessor) -> list[str]:
    names = list(NUMERIC_FEATURES)
    ohe = preprocessor.named_transformers_["cat"].named_steps["encode"]
    for col, categories in zip(CATEGORICAL_FEATURES, ohe.categories_):
        names.extend(f"{col}={c}" for c in categories)
    return names


def run() -> None:
    model = joblib.load(MODEL_ARTIFACT_PATH)
    preprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)

    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_test_raw = test_df[feature_cols]
    X_test = preprocessor.transform(X_test_raw)
    feature_names = _feature_names(preprocessor)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.Series(mean_abs_shap, index=feature_names)
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.iloc[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (impact on churn probability)")
    ax.set_title("Top drivers of predicted churn")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    top5 = importance.head(5)
    lines = [
        "# Explainability Summary\n",
        "## Top factors driving churn predictions (SHAP, held-out test set)\n",
    ]
    for feat, val in top5.items():
        lines.append(f"- `{feat}` (mean impact {val:.3f})")

    lines.append(
        "\n## In plain language, for a non-technical stakeholder\n\n"
        "The model looks at each customer's recent behaviour and flags anyone whose pattern "
        "resembles customers who left before. The strongest signals it uses are:\n\n"
        "- **How long they've been a customer (tenure)** — newer customers churn more; this is "
        "the single biggest driver in the model.\n"
        "- **Whether they've raised a complaint / support ticket** — customers who have had a "
        "service problem are meaningfully more likely to leave.\n"
        "- **How recently and how much they've ordered** — customers with fewer/lower-value orders "
        "and a longer gap since their last purchase are more likely to be flagged.\n"
        "- **Number of registered delivery addresses** — an unexpectedly strong signal in this "
        "data; likely a proxy for account complexity or household/shared usage rather than a "
        "direct cause of churn, and worth validating with the business before acting on it.\n"
        "- Their **preferred product category and device** also shift risk somewhat — e.g. "
        "customers primarily buying mobile-category products churn more in this data than "
        "grocery buyers.\n\n"
        "**Caveat worth flagging to stakeholders:** in this dataset, 'days since last order' has "
        "a counter-intuitive relationship with churn — customers flagged as churned actually have "
        "*fewer* days since their last order on average than retained customers. This likely "
        "reflects how the source system defined/labelled churn (e.g. a snapshot taken at a fixed "
        "point rather than 'no purchase in N days'), not a general truth about e-commerce "
        "customers. This is exactly the kind of finding that needs validating against the "
        "business's actual churn definition before acting on it operationally.\n\n"
        "**Limitations to communicate:** this is trained on a single historical snapshot of "
        "~5,600 customers from one dataset; it will need retraining and re-validation once real "
        "production data with a business-agreed churn definition is available, and predictions "
        "should be treated as prioritisation signals for a retention team, not as an automatic "
        "cutoff for customer treatment.\n"
    )

    (REPORTS_DIR / "explainability_summary.md").write_text("\n".join(lines))
    print(f"Wrote explainability report and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
