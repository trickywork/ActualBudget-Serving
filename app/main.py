from __future__ import annotations

from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.feature_adapter import build_feature_frame, choose_description
from app.runtime import get_backend, warmup_backend
from app.schemas import (
    BatchPredictRequest,
    CategoryScore,
    HealthResponse,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)
from app.telemetry import hardware_string


settings = get_settings()
app = FastAPI(title="ActualBudget Smart Transaction Categorization API", version=settings.code_version)


def _response_from_row(label: str, probabilities: np.ndarray, classes: List[str]) -> PredictResponse:
    ordered_idx = np.argsort(probabilities)[::-1][: settings.top_k]
    top_categories = [
        CategoryScore(category_id=str(classes[idx]), score=round(float(probabilities[idx]), 6))
        for idx in ordered_idx
    ]

    predicted_idx = ordered_idx[0]
    predicted_label = label if label in classes else top_categories[0].category_id
    if predicted_label in classes:
        predicted_idx = classes.index(predicted_label)

    return PredictResponse(
        predicted_category_id=str(predicted_label),
        confidence=round(float(probabilities[predicted_idx]), 6),
        top_categories=top_categories,
        model_version=settings.model_version,
    )


def _predict_many(items: list[PredictRequest]) -> list[PredictResponse]:
    if not items:
        raise ValueError("items must not be empty")
    if any(not choose_description(item) for item in items):
        raise ValueError("transaction_description or merchant_text is required")

    backend = get_backend()
    frame = build_feature_frame(items)
    output = backend.predict(frame)
    return [
        _response_from_row(output.labels[idx], output.probabilities[idx], output.classes)
        for idx in range(len(items))
    ]


@app.on_event("startup")
def _startup() -> None:
    warmup_backend()


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ready=True,
        backend_kind=settings.backend_kind,
        model_version=settings.model_version,
        code_version=settings.code_version,
    )


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    try:
        warmup_backend()
        return HealthResponse(
            status="ready",
            ready=True,
            backend_kind=settings.backend_kind,
            model_version=settings.model_version,
            code_version=settings.code_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/versionz", response_model=VersionResponse)
def versionz() -> VersionResponse:
    backend = get_backend()
    return VersionResponse(
        backend_kind=settings.backend_kind,
        model_version=settings.model_version,
        code_version=settings.code_version,
        model_path=settings.model_path,
        source_model_path=settings.source_model_path,
        providers=backend.providers(),
        hardware=hardware_string(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return _predict_many([request])[0]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: BatchPredictRequest) -> PredictBatchResponse:
    try:
        return PredictBatchResponse(items=_predict_many(request.items))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
