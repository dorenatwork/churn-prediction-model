# EDA Summary (train split only, n=3940)

- Churn rate: **16.8%** (663 churned of 3940) — moderately imbalanced, informs metric choice (PR-AUC/recall over accuracy) and class weighting in training.

## Missing values (numeric features, train split)

- `days_since_last_order`: 5.4% missing
- `orders`: 4.7% missing
- `warehouse_to_home`: 4.7% missing
- `coupon_used`: 4.6% missing
- `tenure_months`: 4.6% missing
- `order_amount_hike_pct`: 4.5% missing
- `hour_spend_on_app`: 4.4% missing

Missingness is consistently ~4-6% across several unrelated columns (Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder) with no obvious pattern tied to churn — treated as missing-at-random and handled with median imputation inside the modelling pipeline (fit on train only).

## Features most correlated with churn (train, |Pearson r|)

- `tenure_months`: r = -0.354
- `complain`: r = +0.246
- `days_since_last_order`: r = -0.159
- `average_order_value`: r = -0.152
- `number_of_device_registered`: r = +0.120

## Categorical feature churn rates

**subscription_type**
- Mobile: 26.9%
- Fashion: 16.8%
- Laptop & Accessory: 10.6%
- Others: 5.2%
- Grocery: 4.1%

**country**
- Tier 3: 22.3%
- Tier 2: 17.8%
- Tier 1: 14.1%

**gender**
- Male: 17.9%
- Female: 15.2%

**marital_status**
- Single: 26.7%
- Divorced: 13.7%
- Married: 12.0%

**preferred_login_device**
- Computer: 20.7%
- Mobile Phone: 15.3%

**preferred_payment_mode**
- Cash on Delivery: 24.6%
- E wallet: 23.0%
- UPI: 17.4%
- Debit Card: 15.7%
- Credit Card: 13.8%

## Data-quality issues found and fixed
- Duplicate/inconsistent category labels: `PreferredLoginDevice` had both 'Phone' and 'Mobile Phone'; `PreferedOrderCat` had both 'Mobile' and 'Mobile Phone'; `PreferredPaymentMode` had both 'COD'/'Cash on Delivery' and 'CC'/'Credit Card'. All folded to a single canonical label before training (see `src/data_prep.py`).
- A small number of duplicate `CustomerID` rows were dropped to avoid the same customer appearing in more than one split.
- No literal order-value or country field exists in the source dataset — adapted to the closest available proxies (see README `Data mapping & assumptions`).
