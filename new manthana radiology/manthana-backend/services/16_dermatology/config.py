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

# Kimi K2.5 (Moonshot) — OpenAI-compatible chat.completions
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")
KIMI_DERMATOLOGY_MODEL = os.getenv("KIMI_DERMATOLOGY_MODEL") or KIMI_MODEL
# Moonshot API allows only {"type": "enabled"} or {"type": "disabled"} for kimi-k2.5
KIMI_DERMATOLOGY_THINKING = os.getenv("KIMI_DERMATOLOGY_THINKING", "enabled").strip().lower()
# httpx/OpenAI client: vision + thinking can be slow
KIMI_DERMATOLOGY_TIMEOUT_SEC = float(os.getenv("KIMI_DERMATOLOGY_TIMEOUT_SEC", "240"))
KIMI_DERMATOLOGY_MAX_RETRIES = int(os.getenv("KIMI_DERMATOLOGY_MAX_RETRIES", "2"))
