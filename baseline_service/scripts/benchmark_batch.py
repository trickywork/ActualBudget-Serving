"""
Simple batch benchmark against the baseline FastAPI service.
"""

import argparse
import json
import statistics
import time
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict_batch")
    parser.add_argument("--input", default="tests/example_batch_request.json")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    payload = json.load(open(args.input))
    n_items = len(payload["items"])
    latencies = []
    for _ in range(args.trials):
        start = time.perf_counter()
        r = requests.post(args.url, json=payload, timeout=30)
        r.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000.0)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    throughput = (n_items * args.trials) / (sum(latencies) / 1000.0)
    print({
        "trials": args.trials,
        "batch_size": n_items,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "throughput_items_per_sec": round(throughput, 2),
    })

if __name__ == "__main__":
    main()
