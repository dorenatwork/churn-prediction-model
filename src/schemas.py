"""Pydantic request/response models for the churn prediction API. See README for the support_tickets -> complain rename rationale."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    tenure_months: float = Field(..., ge=0, description="Months since the customer's first order")
    orders: int = Field(..., ge=0, description="Total number of orders placed")
    average_order_value: float = Field(..., ge=0, description="Average value per order")
    days_since_last_order: float = Field(..., ge=0, description="Days since the most recent order")
    complain: int = Field(
        ..., ge=0, le=1, description="Whether the customer has ever raised a complaint: 0 = no, 1 = yes (binary flag, not a ticket count)"
    )

    # Optional: imputed if omitted, let the model use more signal when available
    satisfaction_score: Optional[float] = Field(None, ge=1, le=5)
    hour_spend_on_app: Optional[float] = Field(None, ge=0)
    number_of_device_registered: Optional[int] = Field(None, ge=0)
    number_of_address: Optional[int] = Field(None, ge=0)
    warehouse_to_home: Optional[float] = Field(None, ge=0, description="Distance from warehouse to customer, a delivery-friction proxy")
    order_amount_hike_pct: Optional[float] = Field(None, description="% increase in order amount vs. last year")
    coupon_used: Optional[int] = Field(None, ge=0)
    subscription_type: Optional[str] = Field(None, description="Preferred order category, used as a subscription/segment proxy")
    country: Optional[str] = Field(None, description="City tier proxy ('Tier 1'/'Tier 2'/'Tier 3') — see README for mapping notes")
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    preferred_login_device: Optional[str] = None
    preferred_payment_mode: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 14,
                "orders": 8,
                "average_order_value": 72.50,
                "days_since_last_order": 63,
                "complain": 1,
            }
        }
    }


class ChurnResponse(BaseModel):
    churn_probability: float
    prediction: Literal["low_risk", "medium_risk", "high_risk"]


class HealthResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
