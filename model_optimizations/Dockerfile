FROM python:3.10-slim

WORKDIR /workspace

COPY model_optimizations/requirements-modelopt.txt /workspace/model_optimizations/requirements-modelopt.txt
RUN pip install --no-cache-dir -r /workspace/model_optimizations/requirements-modelopt.txt

COPY model_optimizations /workspace/model_optimizations
COPY model_input_sample.json /workspace/model_input_sample.json
COPY model_output_sample.json /workspace/model_output_sample.json

CMD ["python", "/workspace/model_optimizations/scripts/benchmark_modelopt.py"]