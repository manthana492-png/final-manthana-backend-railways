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
# HAM10000-style 7-class head on EfficientNet-V2-M (torchvision); upload to MODEL_DIR
HAM_CHECKPOINT_FILENAME = os.getenv("DERM_HAM_CHECKPOINT", "derm_efficientnet_v2m_ham7.pt")
DEVICE = os.getenv("DEVICE", "cpu")

# Comma-separated priority: ham_v2, b4, openrouter (first available wins)
_raw_priority = os.getenv("DERM_CLASSIFIER_PRIORITY", "ham_v2,b4,openrouter").strip().lower()
DERM_CLASSIFIER_PRIORITY = tuple(
    x.strip() for x in _raw_priority.split(",") if x.strip() in ("ham_v2", "b4", "openrouter")
) or ("ham_v2", "b4", "openrouter")

# HAM7 logit order in checkpoint (must match training). Default: HAM10000 folder order.
HAM7_CLASS_ORDER = tuple(
    x.strip()
    for x in os.getenv(
        "DERM_HAM7_CLASS_ORDER",
        "akiec,bcc,bkl,df,mel,nv,vasc",
    ).split(",")
    if x.strip()
)
if len(HAM7_CLASS_ORDER) != 7:
    HAM7_CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

# Grad-CAM on local PyTorch classifiers (best-effort; non-blocking)
DERM_GRADCAM = os.getenv("DERM_GRADCAM", "false").strip().lower() in ("1", "true", "yes")

# Upload / image safety (multipart raw bytes before base64)
DERM_MAX_UPLOAD_BYTES = int(os.getenv("DERM_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
# Guard against decompression bombs / huge frames after decode
DERM_MAX_IMAGE_PIXELS = int(os.getenv("DERM_MAX_IMAGE_PIXELS", str(36_000_000)))  # ~6000×6000
