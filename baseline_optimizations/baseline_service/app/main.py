"""
Baseline FastAPI service for Smart Transaction Categorization.
This service is intentionally simple so it can serve as the CPU baseline on Chameleon.
"""

import os
import time
import socket
import subprocess
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/dummy_pipeline.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dummy-v1")
CODE_VERSION = os.getenv("CODE_VERSION", "dev")
TOP_K = int(os.getenv("TOP_K", "3"))

app = FastAPI(title="Transaction Categorization Baseline", version=CODE_VERSION)
_model = None


class PredictRequest(BaseModel):
    request_id: str = Field(..., description="Client-generated request ID")
    transaction_id: str
    merchant_text: str
    amount: float
    currency: str = "USD"
    transaction_date: str
    account_type: str = "checking"


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]


class Prediction(BaseModel):
    category_id: str
    score: float


class PredictResponse(BaseModel):
    request_id: str
    transaction_id: str
    predictions: List[Prediction]
    model_version: str
    code_version: str
    hostname: str
    latency_ms: float


def _compose_text(item: PredictRequest) -> str:
    """Build a single text field for the simple sklearn pipeline."""
    return (
        f"{item.merchant_text} "
        f"amount_{round(item.amount)} "
        f"currency_{item.currency} "
        f"acct_{item.account_type}"
    )


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def _predict_one(item: PredictRequest) -> PredictResponse:
    model = _load_model()
    start = time.perf_counter()

    text = _compose_text(item)
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
    else:
        # Fall back to a score-based ranking if probability is unavailable.
        decision = model.decision_function([text])[0]
        decision = np.asarray(decision, dtype=float)
        exp = np.exp(decision - np.max(decision))
        probs = exp / exp.sum()

    order = np.argsort(probs)[::-1][:TOP_K]
    preds = [
        Prediction(category_id=classes[idx], score=float(probs[idx]))
        for idx in order
    ]
    latency_ms = (time.perf_counter() - start) * 1000.0

    return PredictResponse(
        request_id=item.request_id,
        transaction_id=item.transaction_id,
        predictions=preds,
        model_version=MODEL_VERSION,
        code_version=CODE_VERSION,
        hostname=socket.gethostname(),
        latency_ms=round(latency_ms, 3),
    )


@app.on_event("startup")
def startup_event():
    # Load the model on startup so readiness reflects real serving state.
    _load_model()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    try:
        _load_model()
        return {"status": "ready", "model_path": MODEL_PATH, "model_version": MODEL_VERSION}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/versionz")
def versionz():
    git_sha = None
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_sha = CODE_VERSION
    return {"model_version": MODEL_VERSION, "code_version": git_sha}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return _predict_one(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(req: BatchPredictRequest):
    try:
        return [_predict_one(item) for item in req.items]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
