"""Baseline FastAPI service for transaction categorization.

Design goals:
1. Get the serving path working end to end first.
2. Expose /predict and /predict_batch clearly.
3. When replacing the real model later, try to only edit model_loader, preprocess, and predict.
4. Leave Prometheus metrics in place so monitoring can be added in phase 2.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .model_loader import LoadedModel, load_model
from .predict import top_k_predictions
from .preprocess import transactions_to_texts
from .schemas import (
    BatchItemResponse,
    ErrorResponse,
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)


STATE: Dict[str, Any] = {
    "loaded_model": None,
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and run a warm-up pass.

    In serving systems, it is common to load the model at startup and do one warm-up pass.
    This block will remain useful later for model-level and system-level experiments as well.
    """
    loaded_model: LoadedModel = load_model()
    # Warm-up so the first real request does not include one-time initialization cost.
    _ = loaded_model.pipeline.predict(["merchant=uber eats acct=credit curr=USD amt_10_50"])
    STATE["loaded_model"] = loaded_model
    STATE["ready"] = True
    yield
    STATE["ready"] = False
    STATE["loaded_model"] = None


app = FastAPI(
    title="Smart Transaction Categorization Serving API",
    version="0.1.0",
    description="Baseline CPU-only FastAPI service for top-k transaction category suggestions.",
    lifespan=lifespan,
)

# This block is expected to stay mostly unchanged; it can be reused directly when
# Prometheus and Grafana are added later.
Instrumentator().instrument(app).expose(app)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = None
    try:
        body = await request.json()
        request_id = body.get("request_id")
    except Exception:
        pass
    error = ErrorResponse(
        request_id=request_id,
        error_code=f"http_{exc.status_code}",
        message=str(exc.detail),
    )
    return JSONResponse(status_code=exc.status_code, content=error.model_dump())


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    loaded = STATE["loaded_model"] is not None
    version = STATE["loaded_model"].model_version if loaded else None
    return HealthResponse(status="ok", model_loaded=loaded, model_version=version)


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="model_not_ready")
    loaded_model: LoadedModel = STATE["loaded_model"]
    return HealthResponse(status="ready", model_loaded=True, model_version=loaded_model.model_version)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not STATE["ready"] or STATE["loaded_model"] is None:
        raise HTTPException(status_code=503, detail="model_not_ready")

    started = time.perf_counter()
    loaded_model: LoadedModel = STATE["loaded_model"]

    texts = transactions_to_texts([request.transaction])
    preds = top_k_predictions(loaded_model, texts, request.top_k)[0]
    latency_ms = (time.perf_counter() - started) * 1000.0

    return PredictResponse(
        request_id=request.request_id,
        transaction_id=request.transaction.transaction_id,
        predictions=preds,
        model_version=loaded_model.model_version,
        code_version=loaded_model.code_version,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    if not STATE["ready"] or STATE["loaded_model"] is None:
        raise HTTPException(status_code=503, detail="model_not_ready")

    started = time.perf_counter()
    loaded_model: LoadedModel = STATE["loaded_model"]

    texts = transactions_to_texts(request.transactions)
    all_preds = top_k_predictions(loaded_model, texts, request.top_k)
    latency_ms = (time.perf_counter() - started) * 1000.0

    results = [
        BatchItemResponse(transaction_id=tx.transaction_id, predictions=preds)
        for tx, preds in zip(request.transactions, all_preds)
    ]

    return PredictBatchResponse(
        request_id=request.request_id,
        model_version=loaded_model.model_version,
        code_version=loaded_model.code_version,
        batch_size=len(request.transactions),
        latency_ms=round(latency_ms, 3),
        results=results,
    )
