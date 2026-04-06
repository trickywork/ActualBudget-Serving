FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY scripts /app/scripts
COPY model_artifacts /app/model_artifacts
COPY model_input_sample.json /app/
COPY model_output_sample.json /app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]