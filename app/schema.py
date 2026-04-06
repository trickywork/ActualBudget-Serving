from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    transaction_description: str = Field(..., min_length=1)
    country: Optional[str] = None
    currency: Optional[str] = None
    amount: Optional[float] = None
    transaction_date: Optional[str] = None
    account_type: Optional[str] = None


class CategoryScore(BaseModel):
    category_id: str
    score: float


class PredictResponse(BaseModel):
    predicted_category_id: str
    confidence: float
    top_categories: List[CategoryScore]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str