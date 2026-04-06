import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from collect_stats import ResourceMonitor

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.joblib")
ONNX_PATH = os.getenv("ONNX_PATH", "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.onnx")
QUANT_PATH = os.getenv("QUANT_PATH", "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.dynamic_quant.onnx")
REQUEST_PATH = os.getenv("REQUEST_PATH", "/workspace/model_optimizations/baseline_service/tests/example_request.json")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/model_optimizations/results"))
TRIALS = int(os.getenv("TRIALS", "120"))
WARMUP = int(os.getenv("WARMUP", "20"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def file_size_mb(path):
    p = Path(path)
    return round(p.stat().st_size / (1024 ** 2), 4) if p.exists() else None

def load_text():
    payload = json.loads(Path(REQUEST_PATH).read_text(encoding="utf-8"))
    desc = payload.get("transaction_description") or payload.get("merchant_text") or ""
    country = payload.get("country", "US")
    currency = payload.get("currency", "USD")
    return f"{desc} country_{country} currency_{currency}".strip()

def ensure_onnx_artifacts():
    if not Path(ONNX_PATH).exists():
        subprocess.run([sys.executable, "/workspace/model_optimizations/scripts/export_sklearn_to_onnx.py"], check=True)
    if not Path(QUANT_PATH).exists():
        subprocess.run([sys.executable, "/workspace/model_optimizations/scripts/quantize_onnx_dynamic.py"], check=True)

def bench_callable(fn, payload, option, artifact_path):
    for _ in range(WARMUP):
        fn(payload)
    mon = ResourceMonitor()
    lats = []
    mon.start()
    for _ in range(TRIALS):
        s = time.perf_counter()
        fn(payload)
        e = time.perf_counter()
        lats.append((e - s) * 1000.0)
    mon.stop()
    return {
        "option": option,
        "batch_size": 1,
        "p50_latency_ms": round(float(np.percentile(lats, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 4),
        "throughput_txns_per_sec": round(float(TRIALS / (sum(lats) / 1000.0)), 4),
        "artifact_size_mb": file_size_mb(artifact_path),
        **mon.summary(),
    }

def main():
    text = load_text()
    payload = [text]
    ensure_onnx_artifacts()

    rows = []

    model = joblib.load(MODEL_PATH)
    rows.append(bench_callable(lambda x: model.predict(x), payload, "joblib_baseline_cpu", MODEL_PATH))

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    rows.append(bench_callable(lambda x: sess.run(None, {inp: np.array(x, dtype=object).reshape(-1, 1)}), payload, "onnx_cpu", ONNX_PATH))

    qsess = ort.InferenceSession(QUANT_PATH, providers=["CPUExecutionProvider"])
    qinp = qsess.get_inputs()[0].name
    rows.append(bench_callable(lambda x: qsess.run(None, {qinp: np.array(x, dtype=object).reshape(-1, 1)}), payload, "onnx_dynamic_quant_cpu", QUANT_PATH))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "artifact_variant_summary.csv", index=False)
    df.to_json(OUTPUT_DIR / "artifact_variant_summary.json", orient="records", indent=2)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
