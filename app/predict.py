import math
import numpy as np
from app.feature_adapter import build_model_text
from app.model_loader import load_model
from app.schemas import PredictRequest, PredictResponse, CategoryScore
from app.settings import MODEL_VERSION, TOP_K


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predict_one(req: PredictRequest) -> PredictResponse:
    model = load_model()
    model_input = [build_model_text(req)]

    pred = model.predict(model_input)[0]

    # 尽量兼容 sklearn pipeline / classifier
    if hasattr(model, "decision_function"):
        scores = model.decision_function(model_input)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf", None), "decision_function"):
        scores = model.named_steps["clf"].decision_function(
            model.named_steps["tfidf"].transform(model_input)
        )
    else:
        raise RuntimeError("Loaded model does not expose decision_function")

    scores = np.array(scores)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    probs = _softmax(scores[0])

    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf", None), "classes_"):
        classes = model.named_steps["clf"].classes_
    else:
        raise RuntimeError("Loaded model does not expose classes_")

    ranked_idx = np.argsort(probs)[::-1][:TOP_K]
    top_categories = [
        CategoryScore(category_id=str(classes[i]), score=round(float(probs[i]), 6))
        for i in ranked_idx
    ]

    top1 = top_categories[0]

    return PredictResponse(
        predicted_category_id=str(pred),
        confidence=top1.score,
        top_categories=top_categories,
        model_version=MODEL_VERSION,
    )