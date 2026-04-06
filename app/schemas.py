from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    transaction_description: Optional[str] = Field(default=None, min_length=1)
    merchant_text: Optional[str] = None

    country: Optional[str] = "US"
    currency: Optional[str] = "USD"

    amount: Optional[float] = None
    transaction_date: Optional[str] = None
    account_type: Optional[str] = None


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(default_factory=list)


class CategoryScore(BaseModel):
    category_id: str
    score: float


class PredictResponse(BaseModel):
    predicted_category_id: str
    confidence: float
    top_categories: List[CategoryScore]
    model_version: str


class PredictBatchResponse(BaseModel):
    items: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    ready: bool
    backend_kind: str
    model_version: str
    code_version: str


class VersionResponse(BaseModel):
    backend_kind: str
    model_version: str
    code_version: str
    model_path: str
    source_model_path: str
    providers: List[str]
    hardware: str
