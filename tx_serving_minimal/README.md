# Minimal Chameleon package for serving

This package is intentionally small. It is meant to restart your workflow from scratch.

## What is included

### 1. `00_chameleon_setup_and_run.ipynb`
Upload this notebook into the Chameleon Jupyter environment and run it there.

It will:
- create a 6-hour lease on `KVM@TACC`
- launch one `m1.medium` VM with `CC-Ubuntu24.04`
- create and attach security groups for ports 22, 8000, 8888, 3000, and 9090
- allocate and attach a floating IP
- print the SSH command for your Mac

This follows the online-evaluation lab pattern of using one `KVM@TACC` VM plus FastAPI/Prometheus/Grafana/Jupyter ports, and it matches your project proposal that a single Chameleon VM should be enough for the lightweight CPU classifier. fileciteturn12file5 fileciteturn12file13

### 2. `baseline_service/`
A simple CPU baseline FastAPI service.
It is the first serving option you should run on Chameleon, because the serving assignment requires a baseline option and measured Chameleon results. fileciteturn12file4 fileciteturn12file15

Contents:
- `app/main.py` — FastAPI service
- `scripts/make_dummy_model.py` — creates a temporary sklearn model for bring-up
- `scripts/generate_requests.py` — creates synthetic JSONL requests
- `scripts/benchmark_single.py` — single-request latency test
- `scripts/benchmark_batch.py` — batch throughput test
- `Dockerfile` — baseline service container

### 3. `model_optimization/`
A separate CPU-only model-optimization module.

Contents:
- `model_optimization_cpu.ipynb` — compares eager PyTorch, compiled PyTorch, ONNX Runtime CPU, and ONNX dynamic quantization in one notebook
- `Dockerfile.modelopt` — Jupyter container for the notebook
- `requirements-modelopt.txt`

This matches the structure of the model-optimization lab, where an ONNX model is benchmarked, then optimized and benchmarked again, and where dynamic quantization is a valid no-calibration option. fileciteturn12file2 fileciteturn12file16

## What you should do first

1. Delete any old lease/server with the same name in Horizon.
2. Upload `00_chameleon_setup_and_run.ipynb` into Chameleon Jupyter.
3. Run the notebook.
4. Copy the printed SSH command and log into the VM from your Mac.
5. Clone your GitHub repo on the VM.
6. Run the baseline service.
7. Run the model-optimization notebook container.

## Why this package is smaller than the previous one

You said you do **not** want to modify the existing lab notebooks.
So this package keeps the lab files out of the repo and only adds:
- one new provisioning notebook
- one baseline-service module
- one model-optimization module

That is enough to start getting Chameleon results, which is what the serving assignment actually gives credit for. fileciteturn12file4
