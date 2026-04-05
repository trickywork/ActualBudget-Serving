"""Prediction helpers shared by single and batch endpoints."""

from __future__ import annotations

from typing import List
import numpy as np

from .model_loader import LoadedModel
from .schemas import PredictionItem


def top_k_predictions(loaded_model: LoadedModel, texts: List[str], top_k: int) -> List[List[PredictionItem]]:
    pipeline = loaded_model.pipeline

    # For models such as LogisticRegression, prefer predict_proba when available.
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)
        classes = list(pipeline.classes_)
        outputs: List[List[PredictionItem]] = []
        for row in probs:
            idxs = np.argsort(row)[::-1][:top_k]
            outputs.append([
                PredictionItem(category_id=str(classes[i]), score=float(row[i]))
                for i in idxs
            ])
        return outputs

    # Fallback logic: if you later switch to a model such as LinearSVC without
    # predict_proba, rank by decision_function and apply a simple min-max normalization.
    scores = pipeline.decision_function(texts)
    classes = list(pipeline.classes_)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    outputs = []
    for row in scores:
        row = np.asarray(row, dtype=float)
        min_v = float(np.min(row))
        max_v = float(np.max(row))
        norm = (row - min_v) / (max_v - min_v + 1e-8)
        idxs = np.argsort(norm)[::-1][:top_k]
        outputs.append([
            PredictionItem(category_id=str(classes[i]), score=float(norm[i]))
            for i in idxs
        ])
    return outputs
