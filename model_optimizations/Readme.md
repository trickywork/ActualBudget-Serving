# model_optimizations

This directory currently contains two serving-related artifacts that map to different experiment rows:

## 1. baseline_http

Artifacts:

- `baseline_service/Dockerfile`
- `baseline_service/app/main.py`
- `baseline_service/tests/example_request.json`
- `baseline_service/tests/example_batch_request.json`

Purpose:

- Containerized FastAPI baseline prediction service
- Exposes `/health`, `/predict`, and `/metrics`
- Used for end-to-end HTTP latency / throughput / error-rate evaluation

## 2. modelopt_cpu

Artifacts:

- `Dockerfile`
- `requirements-modelopt.txt`
- `scripts/benchmark_modelopt.py`
- `notebooks/model_optimization_cpu.ipynb`
- `artifacts/v2_tfidf_linearsvc_model.joblib`

Purpose:

- Model-level benchmark in a controlled harness
- Evaluates the model artifact itself on fixed hardware/runtime
- Produces artifact size, p50/p95 latency, and throughput

## How to run

### Baseline HTTP service

```bash
docker compose -f model_optimizations/docker-compose.yaml up --build baseline_http
```
