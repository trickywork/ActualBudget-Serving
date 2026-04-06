from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.predict import predict_one
from app.settings import MODEL_VERSION

app = FastAPI(title="ActualBudget Serving API", version="1.0.0")

Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_version=MODEL_VERSION)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return predict_one(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))