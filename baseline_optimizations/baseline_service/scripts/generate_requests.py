"""
Generate synthetic JSONL requests for local and remote benchmarking.
"""

import argparse
import json
import random
from datetime import date, timedelta

MERCHANTS = [
    ("UBER TRIP", "transportation"),
    ("UBER EATS", "restaurants"),
    ("WHOLE FOODS", "groceries"),
    ("STARBUCKS", "coffee"),
    ("SHELL", "gas"),
    ("LANDLORD LLC", "rent"),
    ("NETFLIX", "subscriptions"),
    ("AMAZON", "shopping"),
    ("DELTA AIR", "travel"),
    ("DUKE ENERGY", "utilities"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/generated_requests.jsonl")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    start = date(2026, 1, 1)
    with open(args.output, "w") as f:
        for i in range(args.count):
            merchant, _ = random.choice(MERCHANTS)
            item = {
                "request_id": f"req-{i:05d}",
                "transaction_id": f"tx-{i:05d}",
                "merchant_text": f"{merchant} {random.randint(1000, 9999)}",
                "amount": round(random.uniform(3.5, 200.0), 2),
                "currency": "USD",
                "transaction_date": str(start + timedelta(days=random.randint(0, 90))),
                "account_type": random.choice(["checking", "credit"]),
            }
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {args.count} requests to {args.output}")

if __name__ == "__main__":
    main()
