import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def one_request(url: str, payload: dict, timeout: float):
    start = time.perf_counter()
    ok = False
    status = None
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        status = r.status_code
        ok = r.ok
    except Exception:
        pass
    latency_ms = (time.perf_counter() - start) * 1000
    return {"latency_ms": latency_ms, "ok": ok, "status": status}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((p / 100) * (len(values) - 1)))
    return values[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        payloads = json.load(f)

    tasks = []
    start_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i in range(args.requests):
            payload = payloads[i % len(payloads)]
            tasks.append(ex.submit(one_request, args.url, payload, args.timeout))

        results = [f.result() for f in as_completed(tasks)]

    total_sec = time.perf_counter() - start_all

    latencies = [x["latency_ms"] for x in results]
    oks = sum(1 for x in results if x["ok"])
    errors = len(results) - oks

    summary = {
        "total_requests": len(results),
        "successful_requests": oks,
        "failed_requests": errors,
        "error_rate": round(errors / len(results), 6),
        "throughput_rps": round(len(results) / total_sec, 4),
        "p50_latency_ms": round(percentile(latencies, 50), 4),
        "p95_latency_ms": round(percentile(latencies, 95), 4),
        "mean_latency_ms": round(statistics.mean(latencies), 4),
        "concurrency": args.concurrency,
        "url": args.url,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()