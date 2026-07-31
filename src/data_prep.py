"""Load the raw Kaggle export, map columns to our schema, clean it, and split 70/15/15. See README for the full field mapping."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED_DIR,
    RANDOM_STATE,
    RAW_SHEET_NAME,
    RAW_XLSX_PATH,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)

_LOGIN_DEVICE_MAP = {"Phone": "Mobile Phone"}
_ORDER_CAT_MAP = {"Mobile Phone": "Mobile"}
_PAYMENT_MODE_MAP = {"COD": "Cash on Delivery", "CC": "Credit Card"}
_CITY_TIER_MAP = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}

RENAME_MAP = {
    "Tenure": "tenure_months",
    "OrderCount": "orders",
    "CashbackAmount": "average_order_value",
    "DaySinceLastOrder": "days_since_last_order",
    "Complain": "complain",
    "SatisfactionScore": "satisfaction_score",
    "HourSpendOnApp": "hour_spend_on_app",
    "NumberOfDeviceRegistered": "number_of_device_registered",
    "NumberOfAddress": "number_of_address",
    "WarehouseToHome": "warehouse_to_home",
    "OrderAmountHikeFromlastYear": "order_amount_hike_pct",
    "CouponUsed": "coupon_used",
    "PreferedOrderCat": "subscription_type",
    "PreferredLoginDevice": "preferred_login_device",
    "PreferredPaymentMode": "preferred_payment_mode",
    "Gender": "gender",
    "MaritalStatus": "marital_status",
    "CustomerID": "customer_id",
    "Churn": TARGET_COL,
}


def load_raw() -> pd.DataFrame:
    if not RAW_XLSX_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_XLSX_PATH}. Download "
            "'ankitverma2010/ecommerce-customer-churn-analysis-and-prediction' "
            "from Kaggle and place the .xlsx there."
        )
    return pd.read_excel(RAW_XLSX_PATH, sheet_name=RAW_SHEET_NAME)


def clean_and_map(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Fold inconsistent category labels before anything else
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(_LOGIN_DEVICE_MAP)
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(_ORDER_CAT_MAP)
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(_PAYMENT_MODE_MAP)
    df["CityTier"] = df["CityTier"].map(_CITY_TIER_MAP)

    df = df.rename(columns=RENAME_MAP)
    df = df.rename(columns={"CityTier": "country"})

    # Duplicate customer IDs would leak the same customer across splits.
    before = len(df)
    df = df.drop_duplicates(subset="customer_id")
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate customer_id rows", dropped)

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-9

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    # temp_df is val+test; split 50/50 to get 15%/15% of the original data
    val_df, test_df = train_test_split(
        temp_df,
        train_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=temp_df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    raw = load_raw()
    df = clean_and_map(raw)
    train_df, val_df, test_df = split(df)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    logger.info(
        "Saved splits: train=%d (%.1f%% churn) val=%d (%.1f%% churn) test=%d (%.1f%% churn)",
        len(train_df), train_df[TARGET_COL].mean() * 100,
        len(val_df), val_df[TARGET_COL].mean() * 100,
        len(test_df), test_df[TARGET_COL].mean() * 100,
    )


if __name__ == "__main__":
    run()
