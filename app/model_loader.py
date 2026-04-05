"""Model loading utilities.

No matter whether you later switch to a real sklearn model, ONNX Runtime,
or another serving backend, keep the changes concentrated here instead of
spreading them across API handlers.
"""

from __future__ import annotations

import os
import joblib
from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/dummy_tfidf_logreg.joblib")
DEFAULT_MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline-dummy-v1")
DEFAULT_CODE_VERSION = os.getenv("CODE_VERSION", "dev")


@dataclass
class LoadedModel:
    pipeline: Any
    model_path: str
    model_version: str
    code_version: str


def load_model() -> LoadedModel:
    pipeline = joblib.load(DEFAULT_MODEL_PATH)
    return LoadedModel(
        pipeline=pipeline,
        model_path=DEFAULT_MODEL_PATH,
        model_version=DEFAULT_MODEL_VERSION,
        code_version=DEFAULT_CODE_VERSION,
    )
