import json
from pathlib import Path

samples = [
    {
        "transaction_description": "STARBUCKS STORE 1458 NEW YORK NY",
        "country": "US",
        "currency": "USD",
        "amount": 6.45,
        "transaction_date": "2026-04-06",
        "account_type": "credit",
    },
    {
        "transaction_description": "WHOLE FOODS MARKET COLUMBUS OH",
        "country": "US",
        "currency": "USD",
        "amount": 42.10,
        "transaction_date": "2026-04-06",
        "account_type": "checking",
    },
    {
        "transaction_description": "SHELL OIL 57444983902",
        "country": "US",
        "currency": "USD",
        "amount": 38.76,
        "transaction_date": "2026-04-06",
        "account_type": "credit",
    },
]

Path("results").mkdir(exist_ok=True)
with open("results/sample_requests.json", "w") as f:
    json.dump(samples, f, indent=2)