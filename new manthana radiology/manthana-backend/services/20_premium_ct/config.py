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
# Default: MONAI/VISTA3D-HF layout after ``modal run bootstrap_weights`` (see modal_app/bootstrap_weights.py).
_DEFAULT_VISTA = "/models/vista3d/vista3d_pretrained_model/model.safetensors"
VISTA3D_MODEL_PATH = os.getenv("VISTA3D_MODEL_PATH", _DEFAULT_VISTA).strip() or _DEFAULT_VISTA
VISTA3D_FULL_FORWARD = os.getenv("VISTA3D_FULL_FORWARD", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)


def resolve_vista3d_checkpoint_path() -> str:
    """Prefer explicit env path; else first on-disk candidate (bootstrap vs legacy)."""
    explicit = (os.getenv("VISTA3D_MODEL_PATH") or "").strip()
    candidates = [
        explicit,
        _DEFAULT_VISTA,
        "/models/vista3d/model.pt",
        "/models/vista3d/model.safetensors",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return explicit or _DEFAULT_VISTA

