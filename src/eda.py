"""Exploratory data analysis, run on the TRAIN split only to avoid leaking val/test into prep decisions."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    TARGET_COL,
)

sns.set_theme(style="whitegrid")


def run() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")

    lines = ["# EDA Summary (train split only, n={})\n".format(len(train_df))]

    churn_rate = train_df[TARGET_COL].mean()
    lines.append(f"- Churn rate: **{churn_rate:.1%}** ({int(train_df[TARGET_COL].sum())} churned "
                 f"of {len(train_df)}) — moderately imbalanced, informs metric choice (PR-AUC/recall "
                 f"over accuracy) and class weighting in training.\n")

    missing = train_df[NUMERIC_FEATURES].isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    lines.append("## Missing values (numeric features, train split)\n")
    if len(missing):
        for col, pct in missing.items():
            lines.append(f"- `{col}`: {pct:.1%} missing")
    else:
        lines.append("- none")
    lines.append(
        "\nMissingness is consistently ~4-6% across several unrelated columns (Tenure, "
        "WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, "
        "DaySinceLastOrder) with no obvious pattern tied to churn — treated as missing-at-random "
        "and handled with median imputation inside the modelling pipeline (fit on train only).\n"
    )

    # Class balance plot
    fig, ax = plt.subplots(figsize=(4, 4))
    train_df[TARGET_COL].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"], rotation=0)
    ax.set_title("Class balance (train)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=120)
    plt.close(fig)

    # Numeric feature distributions by churn
    numeric_present = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    n_cols = 3
    n_rows = -(-len(numeric_present) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_present):
        sns.kdeplot(data=train_df, x=col, hue=TARGET_COL, common_norm=False, ax=axes[i], warn_singular=False)
        axes[i].set_title(col)
    for j in range(len(numeric_present), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_distributions_by_churn.png", dpi=120)
    plt.close(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = train_df[numeric_present + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    top_corr = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False).head(5)
    lines.append("## Features most correlated with churn (train, |Pearson r|)\n")
    for col, val in top_corr.items():
        signed = corr[TARGET_COL][col]
        lines.append(f"- `{col}`: r = {signed:+.3f}")
    lines.append("")

    lines.append("## Categorical feature churn rates\n")
    for col in CATEGORICAL_FEATURES:
        if col not in train_df.columns:
            continue
        rates = train_df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
        lines.append(f"**{col}**")
        for level, rate in rates.items():
            lines.append(f"- {level}: {rate:.1%}")
        lines.append("")

    lines.append(
        "## Data-quality issues found and fixed\n"
        "- Duplicate/inconsistent category labels: `PreferredLoginDevice` had both 'Phone' and "
        "'Mobile Phone'; `PreferedOrderCat` had both 'Mobile' and 'Mobile Phone'; "
        "`PreferredPaymentMode` had both 'COD'/'Cash on Delivery' and 'CC'/'Credit Card'. "
        "All folded to a single canonical label before training (see `src/data_prep.py`).\n"
        "- A small number of duplicate `CustomerID` rows were dropped to avoid the same customer "
        "appearing in more than one split.\n"
        "- No literal order-value or country field exists in the source dataset — adapted to the "
        "closest available proxies (see README `Data mapping & assumptions`).\n"
    )

    report_path = REPORTS_DIR / "eda_summary.md"
    report_path.write_text("\n".join(lines))
    print(f"Wrote {report_path} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
