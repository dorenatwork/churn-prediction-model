import pandas as pd
import pytest

from src.config import TARGET_COL
from src.data_prep import clean_and_map, split


@pytest.fixture()
def raw_sample():
    return pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4, 4],  # 4 is a duplicate id
            "Churn": [1, 0, 0, 1, 1],
            "Tenure": [1.0, 5.0, None, 10.0, 10.0],
            "PreferredLoginDevice": ["Phone", "Computer", "Mobile Phone", "Phone", "Phone"],
            "CityTier": [1, 2, 3, 1, 1],
            "WarehouseToHome": [10.0, 12.0, 8.0, 20.0, 20.0],
            "PreferredPaymentMode": ["COD", "CC", "UPI", "COD", "COD"],
            "Gender": ["Male", "Female", "Male", "Female", "Female"],
            "HourSpendOnApp": [2.0, 3.0, 1.5, 4.0, 4.0],
            "NumberOfDeviceRegistered": [3, 4, 2, 5, 5],
            "PreferedOrderCat": ["Mobile Phone", "Mobile", "Fashion", "Grocery", "Grocery"],
            "SatisfactionScore": [3, 4, 2, 5, 5],
            "MaritalStatus": ["Single", "Married", "Divorced", "Single", "Single"],
            "NumberOfAddress": [2, 3, 1, 4, 4],
            "Complain": [0, 1, 0, 1, 1],
            "OrderAmountHikeFromlastYear": [15.0, 12.0, 18.0, 20.0, 20.0],
            "CouponUsed": [1.0, 2.0, 0.0, 3.0, 3.0],
            "OrderCount": [2.0, 3.0, 1.0, 5.0, 5.0],
            "DaySinceLastOrder": [3.0, 7.0, 1.0, 15.0, 15.0],
            "CashbackAmount": [100.0, 150.0, 120.0, 200.0, 200.0],
        }
    )


def test_clean_and_map_renames_and_dedupes(raw_sample):
    df = clean_and_map(raw_sample)
    assert TARGET_COL in df.columns
    assert "tenure_months" in df.columns
    assert "average_order_value" in df.columns
    assert "country" in df.columns
    assert df["customer_id"].is_unique
    assert len(df) == 4  # one duplicate customer_id dropped


def test_clean_and_map_folds_duplicate_categories(raw_sample):
    df = clean_and_map(raw_sample)
    assert "Phone" not in df["preferred_login_device"].values
    assert "Mobile Phone" not in df["subscription_type"].values
    assert "COD" not in df["preferred_payment_mode"].values
    assert set(df["country"].unique()) <= {"Tier 1", "Tier 2", "Tier 3"}


def test_split_ratios_and_stratification():
    df = pd.DataFrame(
        {
            "customer_id": range(200),
            TARGET_COL: [1] * 40 + [0] * 160,  # 20% churn
        }
    )
    train_df, val_df, test_df = split(df)
    assert len(train_df) == 140
    assert len(val_df) == 30
    assert len(test_df) == 30
    for part in (train_df, val_df, test_df):
        assert abs(part[TARGET_COL].mean() - 0.20) < 0.05
