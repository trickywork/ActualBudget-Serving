import os
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

ONNX_PATH = os.getenv("ONNX_PATH", "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.onnx")
QUANT_PATH = os.getenv("QUANT_PATH", "/workspace/model_optimizations/artifacts/v2_tfidf_linearsvc_model.dynamic_quant.onnx")


def main():
    Path(QUANT_PATH).parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(ONNX_PATH, QUANT_PATH, weight_type=QuantType.QInt8)
    print(f"Quantized ONNX saved to {QUANT_PATH}")


if __name__ == "__main__":
    main()
