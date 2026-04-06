import os
import time
import socket
import subprocess
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.joblib",
)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
CODE_VERSION = os.getenv("CODE_VERSION", "dev")
TOP_K = int(os.getenv("TOP_K", "3"))

app = FastAPI(title="Transaction Categorization Baseline", version=CODE_VERSION)
_model = None


class PredictRequest(BaseModel):
    # agreed sample fields
    transaction_description: Optional[str] = None
    country: str = "US"
    currency: str = "USD"

    # backward-compatible old fields
    merchant_text: Optional[str] = None
    amount: Optional[float] = None
    account_type: Optional[str] = None


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]


class CategoryScore(BaseModel):
    category_id: str
    score: float


class PredictResponse(BaseModel):
    predicted_category_id: str
    confidence: float
    top_categories: List[CategoryScore]
    model_version: str


def _compose_text(item: PredictRequest) -> str:
    base_text = (item.transaction_description or item.merchant_text or "").strip()
    extras = [
        f"country_{item.country}" if item.country else "",
        f"currency_{item.currency}" if item.currency else "",
    ]
    if item.amount is not None:
        extras.append(f"amount_{round(float(item.amount))}")
    if item.account_type:
        extras.append(f"acct_{item.account_type}")
    return " ".join([base_text] + [x for x in extras if x]).strip()


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def _predict_proba_like(model, text: str):
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
    else:
        decision = model.decision_function([text])[0]
        decision = np.asarray(decision, dtype=float)
        if decision.ndim == 0:
            decision = np.array([decision])
        exp = np.exp(decision - np.max(decision))
        probs = exp / exp.sum()
    return classes, probs


def _predict_one(item: PredictRequest) -> PredictResponse:
    model = _load_model()
    text = _compose_text(item)
    if not text:
        raise HTTPException(status_code=400, detail="transaction_description is required")

    classes, probs = _predict_proba_like(model, text)
    order = np.argsort(probs)[::-1][:TOP_K]
    top_categories = [
        CategoryScore(category_id=str(classes[idx]), score=float(probs[idx]))
        for idx in order
    ]
    best = top_categories[0]

    return PredictResponse(
        predicted_category_id=best.category_id,
        confidence=round(best.score, 6),
        top_categories=top_categories,
        model_version=MODEL_VERSION,
    )


@app.on_event("startup")
def startup_event():
    _load_model()


@app.on_event("startup")
def setup_metrics():
    Instrumentator().instrument(app).expose(app)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "hostname": socket.gethostname()}


@app.get("/readyz")
def readyz():
    try:
        _load_model()
        return {"status": "ready", "model_path": MODEL_PATH, "model_version": MODEL_VERSION}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/versionz")
def versionz():
    git_sha = CODE_VERSION
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass
    return {"model_version": MODEL_VERSION, "code_version": git_sha}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return _predict_one(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=List[PredictResponse])
def predict_batch(req: BatchPredictRequest):
    try:
        return [_predict_one(item) for item in req.items]
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
