import os

SERVICE_NAME = "premium_ct_unified"
PORT = int(os.getenv("PORT", "8018"))
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cuda")
VISTA3D_ENABLED = os.getenv("VISTA3D_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
VISTA3D_MODEL_PATH = os.getenv("VISTA3D_MODEL_PATH", "/models/vista3d/model.pt")
VISTA3D_FULL_FORWARD = os.getenv("VISTA3D_FULL_FORWARD", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)

