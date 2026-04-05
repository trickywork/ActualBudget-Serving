# Baseline Dockerfile for the serving service.
# This version prioritizes stability and reproducibility over aggressive slimming.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=models/dummy_tfidf_logreg.joblib \
    MODEL_VERSION=baseline-dummy-v1 \
    CODE_VERSION=container-baseline

WORKDIR /app

# Some sklearn / numpy wheels on the slim image still depend on basic build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 8000

# For system-level experiments later, you can change the worker count here directly.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
