import os

SERVICE_NAME = "body_xray"
PORT = 8001
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cuda")

# CXR LLM narrative: OpenRouter only (SSOT: repo config/cloud_inference.yaml, role vision_primary / narrative_default).
# OPENROUTER_API_KEY is read by shared llm_router (optional OPENROUTER_API_KEY_2).
XRAY_REQUIRE_LLM_NARRATIVE = os.getenv(
    "XRAY_REQUIRE_LLM_NARRATIVE",
    os.getenv("XRAY_REQUIRE_KIMI_NARRATIVE", "1"),
).strip().lower() in ("1", "true", "yes")
# Back-compat alias for imports: same as XRAY_REQUIRE_LLM_NARRATIVE
XRAY_REQUIRE_KIMI_NARRATIVE = XRAY_REQUIRE_LLM_NARRATIVE
