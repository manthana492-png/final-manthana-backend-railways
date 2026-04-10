import os

SERVICE_NAME = "ct_brain"
PORT = int(os.getenv("PORT", "8017"))
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

# Optional TorchScript heads (deploy-time paths)
CT_BRAIN_SUBTYPE_MODEL_PATH = os.getenv("CT_BRAIN_SUBTYPE_MODEL_PATH", "")
CT_BRAIN_SEGMENTATION_MODEL_PATH = os.getenv("CT_BRAIN_SEGMENTATION_MODEL_PATH", "")

CT_BRAIN_NCC_ENABLED = os.getenv("CT_BRAIN_NCC_ENABLED", "true").lower() in ("1", "true", "yes")
CT_BRAIN_MIDLINE_ENABLED = os.getenv("CT_BRAIN_MIDLINE_ENABLED", "true").lower() in ("1", "true", "yes")
