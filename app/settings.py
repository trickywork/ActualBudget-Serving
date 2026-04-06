import os

MODEL_PATH = os.getenv("MODEL_PATH", "model_artifacts/v2_tfidf_linearsvc_model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2_tfidf_linearsvc")
TOP_K = int(os.getenv("TOP_K", "3"))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))