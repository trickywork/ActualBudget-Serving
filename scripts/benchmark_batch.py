"""Benchmark the batch /predict_batch endpoint."""

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


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests-file", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    lines = Path(args.requests_file).read_text(encoding="utf-8").strip().splitlines()
    single_requests = [json.loads(line) for line in lines]
    transactions = [item["transaction"] for item in single_requests]

    batches = []
    for i, txs in enumerate(chunked(transactions, args.batch_size)):
        if len(batches) >= args.num_batches:
            break
        batches.append({
            "request_id": f"batch-{i}",
            "top_k": 3,
            "transactions": txs,
        })

    latencies = []
    failures = 0
    total_records = 0
    started = time.perf_counter()

    for payload in batches:
        total_records += len(payload["transactions"])
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
    throughput = total_records / total_sec if total_sec > 0 else 0.0

    print("=== Batch Benchmark ===")
    print(f"URL: {args.url}")
    print(f"Batches sent: {len(batches)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total records: {total_records}")
    print(f"Batch failures: {failures}")
    print(f"Batch error rate: {failures / max(1, len(batches)):.4f}")
    print(f"p50 batch latency (ms): {percentile(latencies, 50):.3f}")
    print(f"p95 batch latency (ms): {percentile(latencies, 95):.3f}")
    print(f"avg batch latency (ms): {statistics.mean(latencies) if latencies else 0.0:.3f}")
    print(f"throughput (records/s): {throughput:.3f}")


if __name__ == "__main__":
    main()
