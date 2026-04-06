import json, math, random, time
from pathlib import Path
from statistics import median

import requests
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_config(path='config/system_config.example.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_single_payload(cfg, overrides=None):
    payload = load_json(cfg['request_template_path'])
    if overrides:
        payload.update(overrides)
    return payload


def build_batch_payload(cfg, batch_size):
    one = build_single_payload(cfg)
    return {'instances': [one for _ in range(batch_size)]}


def _post(url, payload, timeout):
    start = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000
    ok = 200 <= r.status_code < 300
    return {'ok': ok, 'status_code': r.status_code, 'latency_ms': elapsed_ms}


def summarize(records, wall_time_s=None, total_requests=None):
    lat = [x['latency_ms'] for x in records if x['ok']]
    if not lat:
        return {
            'success_rate': 0.0,
            'median_latency_ms': None,
            'p95_latency_ms': None,
            'throughput_rps': 0.0,
            'total_requests': len(records),
        }
    lat_sorted = sorted(lat)
    p95_idx = min(len(lat_sorted)-1, math.ceil(0.95 * len(lat_sorted)) - 1)
    reqs = total_requests if total_requests is not None else len(records)
    throughput = reqs / wall_time_s if wall_time_s and wall_time_s > 0 else None
    return {
        'success_rate': round(sum(1 for x in records if x['ok']) / len(records), 4),
        'median_latency_ms': round(median(lat_sorted), 3),
        'p95_latency_ms': round(lat_sorted[p95_idx], 3),
        'throughput_rps': round(throughput, 3) if throughput is not None else None,
        'total_requests': reqs,
    }


def run_fixed_concurrency(base_url, path, payload, concurrency=1, total_requests=50, timeout=10):
    url = base_url.rstrip('/') + path
    records = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_post, url, payload, timeout) for _ in range(total_requests)]
        for fut in as_completed(futs):
            records.append(fut.result())
    wall = time.perf_counter() - start
    return summarize(records, wall_time_s=wall, total_requests=total_requests), records


def run_constant_rate(base_url, path, payload, request_rate, duration_s=20, timeout=10):
    url = base_url.rstrip('/') + path
    interval = 1.0 / request_rate
    records = []
    send_times = []
    t0 = time.perf_counter()
    next_t = t0
    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break
        if now < next_t:
            time.sleep(next_t - now)
        send_times.append(time.perf_counter())
        records.append(_post(url, payload, timeout))
        next_t += interval
    wall = max(time.perf_counter() - t0, 1e-9)
    return summarize(records, wall_time_s=wall, total_requests=len(records)), records


def run_poisson_rate(base_url, path, payload, request_rate, duration_s=20, timeout=10, seed=42):
    random.seed(seed)
    url = base_url.rstrip('/') + path
    records = []
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break
        wait = random.expovariate(request_rate)
        time.sleep(wait)
        if time.perf_counter() - t0 >= duration_s:
            break
        records.append(_post(url, payload, timeout))
    wall = max(time.perf_counter() - t0, 1e-9)
    return summarize(records, wall_time_s=wall, total_requests=len(records)), records
