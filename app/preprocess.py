"""Feature normalization helpers.

The current dummy model only consumes one concatenated text field.
That keeps the serving path simple and lets the baseline run quickly.
When you replace it with the real model later, this file will be one of the
main places to update.
"""

from __future__ import annotations

import re
from typing import Iterable, List
from .schemas import TransactionFeatures


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9*#\-./ ]+")


def normalize_merchant_text(text: str) -> str:
    text = text.lower().strip()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def amount_bucket(amount: float) -> str:
    """Map transaction amount into a coarse bucket to provide a simple numeric signal.

    This is intentionally lightweight and can later be replaced with real numeric features.
    """
    absolute = abs(amount)
    if absolute < 10:
        return "amt_lt_10"
    if absolute < 50:
        return "amt_10_50"
    if absolute < 200:
        return "amt_50_200"
    if absolute < 1000:
        return "amt_200_1000"
    return "amt_ge_1000"


def transaction_to_text(tx: TransactionFeatures) -> str:
    merchant = normalize_merchant_text(tx.merchant_text)
    acct = tx.account_type.lower().strip()
    curr = tx.currency.upper().strip()
    amt = amount_bucket(tx.amount)
    # Build one text feature so the baseline TF-IDF + linear classifier can use it directly.
    return f"merchant={merchant} acct={acct} curr={curr} {amt}"


def transactions_to_texts(transactions: Iterable[TransactionFeatures]) -> List[str]:
    return [transaction_to_text(tx) for tx in transactions]
