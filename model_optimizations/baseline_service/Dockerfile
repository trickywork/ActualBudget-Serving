FROM python:3.10-slim

WORKDIR /app

COPY model_optimizations/baseline_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY model_optimizations/baseline_service/app /app/app
COPY model_optimizations/baseline_service/tests /app/tests
COPY model_optimizations/artifacts /app/model_artifacts

ENV MODEL_PATH=/app/model_artifacts/v2_tfidf_linearsvc_model.joblib
ENV MODEL_VERSION=v1
ENV CODE_VERSION=dev

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
