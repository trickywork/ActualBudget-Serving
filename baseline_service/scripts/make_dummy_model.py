"""
Create a tiny sklearn text classifier that mimics the final project interface.
This is only for serving bring-up and early benchmarking.
"""

import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SAMPLES = [
    ("UBER TRIP amount_23 currency_USD acct_credit", "transportation"),
    ("UBER EATS amount_18 currency_USD acct_credit", "restaurants"),
    ("WHOLE FOODS amount_91 currency_USD acct_checking", "groceries"),
    ("STARBUCKS amount_7 currency_USD acct_credit", "coffee"),
    ("SHELL amount_44 currency_USD acct_credit", "gas"),
    ("CHEVRON amount_57 currency_USD acct_credit", "gas"),
    ("LANDLORD LLC amount_1800 currency_USD acct_checking", "rent"),
    ("SPOTIFY amount_12 currency_USD acct_credit", "subscriptions"),
    ("NETFLIX amount_18 currency_USD acct_credit", "subscriptions"),
    ("AMAZON amount_36 currency_USD acct_credit", "shopping"),
    ("TARGET amount_49 currency_USD acct_credit", "shopping"),
    ("DELTA AIR amount_440 currency_USD acct_credit", "travel"),
    ("MARRIOTT amount_290 currency_USD acct_credit", "travel"),
    ("COMCAST amount_85 currency_USD acct_checking", "utilities"),
    ("DUKE ENERGY amount_133 currency_USD acct_checking", "utilities"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/dummy_pipeline.joblib")
    args = parser.parse_args()

    X = [x for x, _ in SAMPLES]
    y = [y for _, y in SAMPLES]

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    joblib.dump(pipe, args.output)
    print(f"Saved dummy model to {args.output}")

if __name__ == "__main__":
    main()
