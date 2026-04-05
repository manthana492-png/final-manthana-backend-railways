import os

SERVICE_NAME = "oral_cancer"
PORT = 8014
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cpu")  # cpu-only safe; cuda if available

ORAL_CANCER_ENABLED = os.getenv("ORAL_CANCER_ENABLED", "true").lower() == "true"

# Clinical photo — legacy fine-tuned head on EfficientNet-B3 (transformers)
CHECKPOINT_FILENAME = "oral_cancer_finetuned.pt"

# Optional: PyTorch weights compatible with torchvision EfficientNet-V2-M (e.g. ported from public repos)
EFFNET_V2M_CHECKPOINT = os.getenv(
    "ORAL_EFFNET_V2M_CHECKPOINT", "oral_effnet_v2m.pt"
)

# Histopathology — UNI encoder (Hugging Face); optional linear head on disk
UNI_MODEL_ID = os.getenv("UNI_MODEL_ID", "MahmoodLab/UNI")
UNI_HEAD_CHECKPOINT = os.getenv("UNI_HEAD_CHECKPOINT", "uni_oral_linear_head.pt")

# Cloud LLM (vision JSON + long-form narrative): OpenRouter only — models from config/cloud_inference.yaml (role oral_cancer).
# Set OPENROUTER_API_KEY (and optionally CLOUD_INFERENCE_CONFIG_PATH).
