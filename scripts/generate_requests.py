"""Generate representative fake transaction requests for demo and benchmarking."""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path


MERCHANTS = [
    ("UBER EATS 1234", 18.72, "credit"),
    ("UBER *TRIP", 22.15, "credit"),
    ("WHOLE FOODS #102", 87.34, "checking"),
    ("STARBUCKS 0912", 6.48, "credit"),
    ("SHELL OIL 5744", 41.02, "credit"),
    ("LANDLORD LLC", 1450.00, "checking"),
    ("AMZN MKTPLACE PMTS", 29.99, "credit"),
    ("NETFLIX.COM", 15.49, "credit"),
    ("CVS PHARMACY", 23.17, "credit"),
    ("VERIZON", 72.50, "checking"),
]


def random_date() -> str:
    start = date(2026, 1, 1)
    delta = random.randint(0, 90)
    return (start + timedelta(days=delta)).isoformat()


def make_record(i: int) -> dict:
    merchant, amount, account_type = random.choice(MERCHANTS)
    amount = round(amount * random.uniform(0.85, 1.15), 2)
    merchant_variant = merchant
    if random.random() < 0.35:
        merchant_variant += f" {random.randint(100, 9999)}"
    return {
        "request_id": f"req-{i}",
        "top_k": 3,
        "transaction": {
            "transaction_id": f"tx-{i}",
            "merchant_text": merchant_variant,
            "amount": amount,
            "currency": "USD",
            "transaction_date": random_date(),
            "account_type": account_type,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            f.write(json.dumps(make_record(i)) + "\n")
    print(f"Wrote {args.num_samples} request records to {path}")


if __name__ == "__main__":
    main()
