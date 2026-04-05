"""
Dermatology service config.

Cloud LLM: OpenRouter only — models from config/cloud_inference.yaml (role dermatology).
Secrets: OPENROUTER_API_KEY (optional OPENROUTER_API_KEY_2).
"""

import os

SERVICE_NAME = "dermatology"
PORT = 8016
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
CHECKPOINT_FILENAME = "derm_efficientnet_b4.pt"
DEVICE = os.getenv("DEVICE", "cpu")

# Upload / image safety (multipart raw bytes before base64)
DERM_MAX_UPLOAD_BYTES = int(os.getenv("DERM_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
# Guard against decompression bombs / huge frames after decode
DERM_MAX_IMAGE_PIXELS = int(os.getenv("DERM_MAX_IMAGE_PIXELS", str(36_000_000)))  # ~6000×6000
