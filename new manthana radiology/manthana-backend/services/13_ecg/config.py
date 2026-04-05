import os

SERVICE_NAME = "ecg"
PORT = 8013
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cpu")  # ECG runs on CPU

# Cloud LLM: OPENROUTER_API_KEY + CLOUD_INFERENCE_CONFIG_PATH (repo config/cloud_inference.yaml).
