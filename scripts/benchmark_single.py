"""Benchmark the single-request /predict endpoint.

When you later run formal experiments on Chameleon, this script should remain part
of the baseline measurement workflow.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, round((p / 100.0) * (len(values) - 1)))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests-file", required=True)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    lines = Path(args.requests_file).read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines[: args.num_requests]]

    latencies = []
    failures = 0
    started = time.perf_counter()

    for payload in payloads:
        t0 = time.perf_counter()
        try:
            resp = requests.post(args.url, json=payload, timeout=args.timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)
            if resp.status_code >= 400:
                failures += 1
        except requests.RequestException:
            failures += 1

    total_sec = time.perf_counter() - started
    throughput = len(payloads) / total_sec if total_sec > 0 else 0.0

    print("=== Single Request Benchmark ===")
    print(f"URL: {args.url}")
    print(f"Requests sent: {len(payloads)}")
    print(f"Failures: {failures}")
    print(f"Error rate: {failures / max(1, len(payloads)):.4f}")
    print(f"p50 latency (ms): {percentile(latencies, 50):.3f}")
    print(f"p95 latency (ms): {percentile(latencies, 95):.3f}")
    print(f"p99 latency (ms): {percentile(latencies, 99):.3f}")
    print(f"avg latency (ms): {statistics.mean(latencies) if latencies else 0.0:.3f}")
    print(f"throughput (req/s): {throughput:.3f}")


if __name__ == "__main__":
    main()
