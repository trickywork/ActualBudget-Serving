import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
REQUEST_PATH = os.getenv("REQUEST_PATH", "/workspace/model_optimizations/baseline_service/tests/example_request.json")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/model_optimizations/results"))
CONCURRENCY = [int(x) for x in os.getenv("CONCURRENCY", "1,4,8").split(",") if x.strip()]
REQUESTS_PER_STAGE = int(os.getenv("REQUESTS_PER_STAGE", "80"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def send_one(session, payload):
    start = time.perf_counter()
    try:
        r = session.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        latency_ms = (time.perf_counter() - start) * 1000.0
        ok = r.status_code == 200
        return {"ok": ok, "status": r.status_code, "latency_ms": latency_ms}
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": False, "status": 0, "latency_ms": latency_ms}

def run_stage(payload, concurrency):
    session = requests.Session()
    results = []
    stage_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(send_one, session, payload) for _ in range(REQUESTS_PER_STAGE)]
        for f in as_completed(futures):
            results.append(f.result())
    stage_s = time.perf_counter() - stage_start
    lats = [x["latency_ms"] for x in results]
    ok_cnt = sum(1 for x in results if x["ok"])
    err_rate = 1 - (ok_cnt / len(results))
    return {
        "concurrency_tested": concurrency,
        "p50_latency_ms": round(float(np.percentile(lats, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 4),
        "throughput_rps": round(len(results) / stage_s, 4),
        "error_rate": round(float(err_rate), 6),
    }

def main():
    with open(REQUEST_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for c in CONCURRENCY:
        row = run_stage(payload, c)
        row["option"] = "baseline_http"
        row["endpoint_url"] = f"{BASE_URL}/predict"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "baseline_http_summary.csv", index=False)
    df.to_json(OUTPUT_DIR / "baseline_http_summary.json", orient="records", indent=2)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
