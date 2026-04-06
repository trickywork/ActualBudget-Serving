
from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_config(path: str = "config/infra_config.example.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_single_payload() -> Dict[str, Any]:
    return {
        "request_id": "req-infra-001",
        "transaction_id": "txn-infra-001",
        "merchant_text": "STARBUCKS STORE 1458 NEW YORK NY",
        "amount": 6.45,
        "currency": "USD",
        "transaction_date": "2026-04-06",
        "account_type": "checking",
    }


def post_json(url: str, payload: Dict[str, Any], timeout: float = 10.0) -> Tuple[float, int, str]:
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return latency_ms, resp.status_code, resp.text[:200]
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return latency_ms, -1, str(e)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def summarize_results(latencies_ms: List[float], status_codes: List[int], wall_s: float) -> Dict[str, Any]:
    total = len(latencies_ms)
    ok = sum(1 for s in status_codes if 200 <= s < 300)
    error_rate = 0.0 if total == 0 else (total - ok) / total
    return {
        "count": total,
        "success_count": ok,
        "error_rate": round(error_rate, 4),
        "median_latency_ms": round(_percentile(latencies_ms, 50), 3),
        "p95_latency_ms": round(_percentile(latencies_ms, 95), 3),
        "p99_latency_ms": round(_percentile(latencies_ms, 99), 3),
        "throughput_rps": round(total / wall_s, 3) if wall_s > 0 else math.nan,
    }


def run_fixed_concurrency(base_url: str, predict_path: str, payload: Dict[str, Any],
                          concurrency: int, total_requests: int = 80,
                          timeout: float = 10.0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    url = base_url.rstrip("/") + predict_path
    details: List[Dict[str, Any]] = []
    latencies: List[float] = []
    statuses: List[int] = []

    def _task(i: int):
        latency_ms, status_code, body = post_json(url, payload, timeout=timeout)
        return {"request_idx": i, "latency_ms": latency_ms, "status_code": status_code, "body": body}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_task, i) for i in range(total_requests)]
        for fut in as_completed(futures):
            row = fut.result()
            details.append(row)
            latencies.append(row["latency_ms"])
            statuses.append(row["status_code"])
    wall_s = time.perf_counter() - start
    return summarize_results(latencies, statuses, wall_s), details


def run_request_rate_profile(base_url: str, predict_path: str, payload: Dict[str, Any],
                             request_rate: float, duration_s: float = 20.0,
                             distribution: str = "constant", timeout: float = 10.0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    url = base_url.rstrip("/") + predict_path
    details: List[Dict[str, Any]] = []
    latencies: List[float] = []
    statuses: List[int] = []

    t0 = time.perf_counter()
    next_send = t0
    sent = 0
    while time.perf_counter() - t0 < duration_s:
        now = time.perf_counter()
        if now < next_send:
            time.sleep(next_send - now)
        latency_ms, status_code, body = post_json(url, payload, timeout=timeout)
        sent += 1
        details.append({"request_idx": sent, "latency_ms": latency_ms, "status_code": status_code, "body": body})
        latencies.append(latency_ms)
        statuses.append(status_code)

        if distribution == "constant":
            interval = 1.0 / request_rate
        elif distribution == "poisson":
            interval = random.expovariate(request_rate)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        next_send = max(next_send + interval, time.perf_counter())

    wall_s = time.perf_counter() - t0
    summary = summarize_results(latencies, statuses, wall_s)
    summary["request_rate_target"] = request_rate
    summary["distribution"] = distribution
    return summary, details


def wait_until_ready(base_url: str, ready_path: str = "/readyz",
                     timeout_s: float = 60.0, poll_interval_s: float = 1.0) -> Dict[str, Any]:
    url = base_url.rstrip("/") + ready_path
    start = time.perf_counter()
    attempts = 0
    while time.perf_counter() - start < timeout_s:
        attempts += 1
        try:
            resp = requests.get(url, timeout=3)
            if 200 <= resp.status_code < 300:
                return {
                    "ready": True,
                    "ready_after_s": round(time.perf_counter() - start, 3),
                    "attempts": attempts,
                    "status_code": resp.status_code,
                }
        except Exception:
            pass
        time.sleep(poll_interval_s)
    return {
        "ready": False,
        "ready_after_s": round(time.perf_counter() - start, 3),
        "attempts": attempts,
        "status_code": None,
    }


def measure_first_request(base_url: str, predict_path: str, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    url = base_url.rstrip("/") + predict_path
    latency_ms, status_code, body = post_json(url, payload, timeout=timeout)
    return {
        "first_request_latency_ms": round(latency_ms, 3),
        "status_code": status_code,
        "body_preview": body,
    }
