import os

SERVICE_NAME = "oral_cancer"
PORT = 8014
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cpu")  # cpu-only safe; cuda if available

ORAL_CANCER_ENABLED = os.getenv("ORAL_CANCER_ENABLED", "true").lower() == "true"

# Clinical photo — legacy fine-tuned head on EfficientNet-B3 (transformers)
CHECKPOINT_FILENAME = "oral_cancer_finetuned.pt"

# Optional: PyTorch weights compatible with torchvision EfficientNet-V2-M (e.g. Apache-2.0 oral checkpoints)
EFFNET_V2M_CHECKPOINT = os.getenv(
    "ORAL_EFFNET_V2M_CHECKPOINT", "oral_effnet_v2m.pt"
)

# Clinical photo order: unset/empty = V2-M before B3 when both weight files exist (production default).
# "0"/"false"/"no" = legacy B3-first. "1"/"true"/"yes" = force V2-M before B3 when V2-M weights exist.
ORAL_PREFER_V2M = os.getenv("ORAL_PREFER_V2M", "").strip()

# V2-M head width must match the checkpoint (2 = binary normal vs malignant, 3 = Normal/OPMD/OSCC).
# Secret files often set ORAL_V2M_NUM_CLASSES= (empty); getenv returns "" and int("") crashes — treat blank as unset.
_v2m_cls = (os.getenv("ORAL_V2M_NUM_CLASSES") or "").strip()
ORAL_V2M_NUM_CLASSES = int(_v2m_cls) if _v2m_cls else 3
if ORAL_V2M_NUM_CLASSES not in (2, 3):
    ORAL_V2M_NUM_CLASSES = 3

# When ORAL_V2M_NUM_CLASSES=2: fraction of malignant mass assigned to OPMD (rest to OSCC-suspicious).
_v2m_frac = (os.getenv("ORAL_V2M_BINARY_OPMD_FRACTION") or "").strip()
ORAL_V2M_BINARY_OPMD_FRACTION = float(_v2m_frac) if _v2m_frac else 0.45
ORAL_V2M_BINARY_OPMD_FRACTION = min(0.95, max(0.05, ORAL_V2M_BINARY_OPMD_FRACTION))

# Histopathology — UNI encoder (Hugging Face); optional linear head on disk
UNI_MODEL_ID = os.getenv("UNI_MODEL_ID", "MahmoodLab/UNI")
UNI_HEAD_CHECKPOINT = os.getenv("UNI_HEAD_CHECKPOINT", "uni_oral_linear_head.pt")

# Cloud LLM (vision JSON + long-form narrative): OpenRouter only — models from config/cloud_inference.yaml (role oral_cancer).
# Set OPENROUTER_API_KEY (and optionally CLOUD_INFERENCE_CONFIG_PATH).
