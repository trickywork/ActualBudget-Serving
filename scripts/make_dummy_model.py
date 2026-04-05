"""Build a small dummy sklearn pipeline artifact for serving development.

Purpose:
- Let the serving path run before the training team delivers the real model
- Produce reasonable top-k outputs for Swagger, demos, and benchmarks
- Be directly replaceable later by the real model artifact
"""

from __future__ import annotations

from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


OUTPUT_PATH = Path("models/dummy_tfidf_logreg.joblib")


def build_training_data():
    """Construct a small, interpretable training set.

    This is not meant to maximize real accuracy. Its only purpose is to make the
    service output look realistic enough and map different merchant patterns to
    plausible categories.
    """
    rows = [
        ("merchant=uber eats acct=credit curr=USD amt_10_50", "restaurants"),
        ("merchant=doordash acct=credit curr=USD amt_10_50", "restaurants"),
        ("merchant=chipotle acct=credit curr=USD amt_10_50", "restaurants"),
        ("merchant=starbucks acct=credit curr=USD amt_lt_10", "coffee"),
        ("merchant=dunkin acct=credit curr=USD amt_lt_10", "coffee"),
        ("merchant=whole foods acct=credit curr=USD amt_50_200", "groceries"),
        ("merchant=kroger acct=checking curr=USD amt_50_200", "groceries"),
        ("merchant=trader joes acct=checking curr=USD amt_10_50", "groceries"),
        ("merchant=shell oil acct=credit curr=USD amt_50_200", "gas"),
        ("merchant=bp gas acct=credit curr=USD amt_10_50", "gas"),
        ("merchant=chevron acct=credit curr=USD amt_10_50", "gas"),
        ("merchant=uber trip acct=credit curr=USD amt_10_50", "transportation"),
        ("merchant=lyft acct=credit curr=USD amt_10_50", "transportation"),
        ("merchant=amtrak acct=credit curr=USD amt_50_200", "transportation"),
        ("merchant=verizon acct=checking curr=USD amt_50_200", "utilities"),
        ("merchant=att bill pay acct=checking curr=USD amt_50_200", "utilities"),
        ("merchant=columbia gas acct=checking curr=USD amt_50_200", "utilities"),
        ("merchant=landlord llc acct=checking curr=USD amt_ge_1000", "rent"),
        ("merchant=apartment rent acct=checking curr=USD amt_ge_1000", "rent"),
        ("merchant=property mgmt acct=checking curr=USD amt_ge_1000", "rent"),
        ("merchant=amazon mktplace acct=credit curr=USD amt_10_50", "shopping"),
        ("merchant=target acct=credit curr=USD amt_10_50", "shopping"),
        ("merchant=best buy acct=credit curr=USD amt_50_200", "shopping"),
        ("merchant=netflix acct=credit curr=USD amt_10_50", "subscriptions"),
        ("merchant=spotify acct=credit curr=USD amt_lt_10", "subscriptions"),
        ("merchant=apple com bill acct=credit curr=USD amt_10_50", "subscriptions"),
        ("merchant=walgreens acct=credit curr=USD amt_10_50", "health"),
        ("merchant=cvs pharmacy acct=credit curr=USD amt_10_50", "health"),
        ("merchant=unitedhealth acct=checking curr=USD amt_50_200", "health"),
    ]
    texts = [x for x, _ in rows]
    labels = [y for _, y in rows]
    return texts, labels


def main() -> None:
    texts, labels = build_training_data()
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(texts, labels)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Saved dummy model to: {OUTPUT_PATH.resolve()}")
    print(f"Classes: {list(pipeline.classes_)}")


if __name__ == "__main__":
    main()
