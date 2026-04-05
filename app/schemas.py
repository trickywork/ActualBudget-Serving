"""Pydantic request / response schemas.

The main goal here is to make the contract explicit first. Even if the training
or data team later changes features, most updates should stay limited to this
file and preprocess.py instead of requiring a full rewrite of the service.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TransactionFeatures(BaseModel):
    """Single transaction input.

    The fields here follow the online feature shape from the project proposal:
    merchant text + amount + date + account context + currency.
    If the training team adds new fields later, they can be extended here.
    """

    transaction_id: str = Field(..., description="Unique transaction identifier")
    merchant_text: str = Field(..., description="Original or normalized merchant/payee text")
    amount: float = Field(..., description="Transaction amount in transaction currency")
    currency: str = Field(default="USD", description="Currency code, e.g. USD")
    transaction_date: str = Field(..., description="ISO date string: YYYY-MM-DD")
    account_type: str = Field(default="checking", description="checking / credit / cash / savings")


class PredictRequest(BaseModel):
    request_id: str = Field(..., description="Caller-generated request identifier")
    top_k: int = Field(default=3, ge=1, le=10, description="How many category suggestions to return")
    transaction: TransactionFeatures


class PredictBatchRequest(BaseModel):
    request_id: str = Field(..., description="Batch request identifier")
    top_k: int = Field(default=3, ge=1, le=10)
    transactions: List[TransactionFeatures] = Field(..., min_length=1, max_length=512)


class PredictionItem(BaseModel):
    category_id: str
    score: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    transaction_id: str
    predictions: List[PredictionItem]
    model_version: str
    code_version: str
    latency_ms: float


class BatchItemResponse(BaseModel):
    transaction_id: str
    predictions: List[PredictionItem]


class PredictBatchResponse(BaseModel):
    request_id: str
    model_version: str
    code_version: str
    batch_size: int
    latency_ms: float
    results: List[BatchItemResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ErrorResponse(BaseModel):
    request_id: Optional[str] = None
    error_code: str
    message: str
