FROM python:3.10-slim

WORKDIR /app

COPY batch_pipeline.py /app/
COPY ingest.py /app/
COPY data_generator.py /app/
COPY online_features.py /app/

RUN pip install --no-cache-dir pandas pyarrow huggingface_hub

CMD ["python3", "/app/batch_pipeline.py"]
