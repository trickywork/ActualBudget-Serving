from app.schemas import PredictRequest


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_model_text(req: PredictRequest) -> str:
    parts = [normalize_text(req.transaction_description)]

    if req.country:
        parts.append(f"country_{req.country.lower()}")

    if req.currency:
        parts.append(f"currency_{req.currency.lower()}")

    if req.account_type:
        parts.append(f"account_{req.account_type.lower()}")

    return " ".join(parts)