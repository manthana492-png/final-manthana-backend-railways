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

# Vision LLM fallback (clinical interpretation when no local classifier succeeds)
CLAUDE_ORAL_MODEL = os.getenv("CLAUDE_ORAL_MODEL", "claude-sonnet-4-20250514")

# Kimi / narrative — same resolution order as other modalities
KIMI_ORAL_CANCER_MODEL = (
    os.getenv("KIMI_ORAL_CANCER_MODEL")
    or os.getenv("KIMI_ORAL_MODEL")
    or os.getenv("KIMI_MODEL", "moonshot-v1-8k")
).strip()

# Long-form screening report (Kimi → Anthropic); separate from vision JSON model
ANTHROPIC_ORAL_NARRATIVE_MODEL = os.getenv(
    "ANTHROPIC_ORAL_NARRATIVE_MODEL", "claude-3-5-sonnet-20241022"
).strip()
