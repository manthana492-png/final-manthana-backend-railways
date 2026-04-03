import os

SERVICE_NAME = "body_xray"
PORT = 8001
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cuda")

# Kimi narrative (Moonshot OpenAI-compatible API)
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")
XRAY_REQUIRE_KIMI_NARRATIVE = os.getenv("XRAY_REQUIRE_KIMI_NARRATIVE", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
