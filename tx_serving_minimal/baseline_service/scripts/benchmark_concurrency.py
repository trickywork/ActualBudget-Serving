import argparse
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


DEFAULT_PAYLOAD = {
    "request_id": "req-bench",
    "transaction_id": "tx-bench",
    "merchant_text": "UBER EATS 1234",
    "amount": 18.72,
    "currency": "USD",
    "transaction_date": "2026-03-25",
    "account_type": "credit",
}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def worker(url, duration_sec, timeout_sec, payload, stats, lock):
    session = requests.Session()
    end_time = time.time() + duration_sec

    local_latencies = []
    local_success = 0
    local_errors = 0

    while time.time() < end_time:
        start = time.perf_counter()
        try:
            resp = session.post(url, json=payload, timeout=timeout_sec)
            latency_ms = (time.perf_counter() - start) * 1000.0
            local_latencies.append(latency_ms)

            if 200 <= resp.status_code < 300:
                local_success += 1
            else:
                local_errors += 1
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            local_latencies.append(latency_ms)
            local_errors += 1

    with lock:
        stats["latencies_ms"].extend(local_latencies)
        stats["success"] += local_success
        stats["errors"] += local_errors


def main():
    parser = argparse.ArgumentParser(description="Concurrent benchmark for /predict endpoint")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    stats = {
        "latencies_ms": [],
        "success": 0,
        "errors": 0,
    }
    lock = threading.Lock()

    wall_start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                worker,
                args.url,
                args.duration,
                args.timeout,
                DEFAULT_PAYLOAD,
                stats,
                lock,
            )
            for _ in range(args.concurrency)
        ]
        for f in as_completed(futures):
            f.result()

    wall_elapsed = time.time() - wall_start

    total_requests = stats["success"] + stats["errors"]
    error_rate = (stats["errors"] / total_requests) if total_requests > 0 else 0.0
    throughput_rps = total_requests / wall_elapsed if wall_elapsed > 0 else 0.0

    p50 = percentile(stats["latencies_ms"], 50)
    p95 = percentile(stats["latencies_ms"], 95)
    mean_ms = statistics.mean(stats["latencies_ms"]) if stats["latencies_ms"] else None

    result = {
        "url": args.url,
        "concurrency": args.concurrency,
        "duration_sec": args.duration,
        "total_requests": total_requests,
        "success": stats["success"],
        "errors": stats["errors"],
        "error_rate": round(error_rate, 4),
        "throughput_rps": round(throughput_rps, 2),
        "p50_ms": round(p50, 3) if p50 is not None else None,
        "p95_ms": round(p95, 3) if p95 is not None else None,
        "mean_ms": round(mean_ms, 3) if mean_ms is not None else None,
    }

    print(result)


if __name__ == "__main__":
    main()