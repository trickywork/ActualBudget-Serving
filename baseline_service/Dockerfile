FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

ENV MODEL_PATH=models/dummy_pipeline.joblib
ENV MODEL_VERSION=dummy-v1
ENV CODE_VERSION=baseline-fastapi
ENV TOP_K=3

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
