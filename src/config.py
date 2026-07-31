"""Shared paths, constants and the target feature schema for the churn project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Stratified on the target
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

TARGET_COL = "churned"
RAW_XLSX_PATH = DATA_RAW_DIR / "E Commerce Dataset.xlsx"
RAW_SHEET_NAME = "E Comm"

# See README "Data mapping & assumptions" for how each field maps to the source dataset.
NUMERIC_FEATURES = [
    "tenure_months",
    "orders",
    "average_order_value",
    "days_since_last_order",
    "complain",
    "satisfaction_score",
    "hour_spend_on_app",
    "number_of_device_registered",
    "number_of_address",
    "warehouse_to_home",
    "order_amount_hike_pct",
    "coupon_used",
]

CATEGORICAL_FEATURES = [
    "subscription_type",
    "country",
    "gender",
    "marital_status",
    "preferred_login_device",
    "preferred_payment_mode",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Required by the /predict payload; other fields are optional.
REQUIRED_API_FIELDS = [
    "tenure_months",
    "orders",
    "average_order_value",
    "days_since_last_order",
    "complain",
]

MLFLOW_EXPERIMENT_NAME = "ecommerce-churn"
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "ecommerce-churn-classifier"

MODEL_ARTIFACT_PATH = MODELS_DIR / "model.joblib"
PREPROCESSOR_ARTIFACT_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_ARTIFACT_PATH = MODELS_DIR / "model_metadata.json"
