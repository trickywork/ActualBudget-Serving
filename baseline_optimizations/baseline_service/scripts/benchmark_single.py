"""
Simple single-request benchmark against the baseline FastAPI service.
"""

import argparse
import json
import statistics
import time
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--input", default="tests/example_request.json")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    payload = json.load(open(args.input))
    latencies = []
    for _ in range(args.trials):
        start = time.perf_counter()
        r = requests.post(args.url, json=payload, timeout=30)
        r.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000.0)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    print({"trials": args.trials, "p50_ms": round(p50, 3), "p95_ms": round(p95, 3)})

if __name__ == "__main__":
    main()
