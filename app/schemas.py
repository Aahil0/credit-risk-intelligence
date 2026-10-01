from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

Status = Annotated[int, Field(ge=-2, le=9)]
Money = Annotated[int, Field(ge=-10_000_000, le=10_000_000)]
Payment = Annotated[int, Field(ge=0, le=10_000_000)]


class Applicant(BaseModel):
    """Arrays ordered September -> April; money is in New Taiwan dollars."""
    model_config = ConfigDict(extra='forbid', strict=True, json_schema_extra={
        'examples': [{'credit_limit': 200000, 'repayment_status': [0, 0, 0, 0, 0, 0],
                      'bill_amounts': [45000, 43000, 40000, 38000, 35000, 33000],
                      'payment_amounts': [5000, 5000, 5000, 5000, 5000, 5000]}]})
    credit_limit: Annotated[int, Field(ge=1, le=10_000_000)]
    repayment_status: Annotated[list[Status], Field(min_length=6, max_length=6)]
    bill_amounts: Annotated[list[Money], Field(min_length=6, max_length=6)]
    payment_amounts: Annotated[list[Payment], Field(min_length=6, max_length=6)]


class RiskFactor(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    feature: str
    label: str
    contribution: float
    direction: Literal['increases_risk', 'decreases_risk']
    transformed_feature_value: float
    feature_value: float
    value_unit: str
    indicator_active: bool | None = None
    indicator_code: int | None = None


class Prediction(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    probability_of_default: Annotated[float, Field(ge=0, le=1)]
    risk_category: Literal['LOW','MEDIUM','HIGH']
    flagged_for_review: bool
    important_risk_factors: list[RiskFactor]
    explanation_unit: str
    explanation_base_value: float
    explanation_total_contribution: float
    model: str
    calibration: str
    decision_threshold: Annotated[float, Field(ge=0, le=1)]
    warnings: list[str]
    notice: str


class Health(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    status: Literal['ok']
    model: str
    version: str
    educational_only: Literal[True]
