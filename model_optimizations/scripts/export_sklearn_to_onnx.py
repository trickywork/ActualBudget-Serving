import os
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.joblib")
ONNX_PATH = os.getenv("ONNX_PATH", "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.onnx")


def main():
    model = joblib.load(MODEL_PATH)
    onx = convert_sklearn(
        model,
        initial_types=[("input", StringTensorType([None, 1]))],
        target_opset=17,
    )
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(ONNX_PATH, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Exported ONNX to {ONNX_PATH}")


if __name__ == "__main__":
    main()
