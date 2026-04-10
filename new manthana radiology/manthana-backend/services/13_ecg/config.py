import os

SERVICE_NAME = "ecg"
PORT = 8013
DEVICE = os.getenv("DEVICE", "cpu")  # ECG runs on CPU

# Cloud LLM: OPENROUTER_API_KEY + CLOUD_INFERENCE_CONFIG_PATH (repo config/cloud_inference.yaml).
# No NN checkpoints on Modal volume: Manthana-ECG-Engine (heuristics) + neurokit2 + narrative_ecg prompts.

# Optional vendored PhysioNet digitiser repo root (contains upstream digitise entrypoint)
ECG_DIGITISER_REPO_ROOT = os.getenv("ECG_DIGITISER_REPO_ROOT", "").strip()

ECG_HEURISTIC_FALLBACK = os.getenv("ECG_HEURISTIC_FALLBACK", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "",
)

# Per-stage timeouts (seconds)
ECG_DIGITISE_TIMEOUT_SEC = float(os.getenv("ECG_DIGITISE_TIMEOUT_SEC", "120"))

# Image quality gate (JPEG/PNG before digitization)
ECG_IMAGE_QUALITY_GATE = os.getenv("ECG_IMAGE_QUALITY_GATE", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "",
)
ECG_IMAGE_MIN_SHORT_EDGE = int(os.getenv("ECG_IMAGE_MIN_SHORT_EDGE", "480"))
ECG_IMAGE_BLUR_VARIANCE_MIN = float(os.getenv("ECG_IMAGE_BLUR_VARIANCE_MIN", "15.0"))

# Semantic version for API consumers
ECG_PIPELINE_VERSION = os.getenv("ECG_PIPELINE_VERSION", "heuristic-prompt-1.0.0").strip()
