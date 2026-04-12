"""Deploy unified Premium CT service (full VISTA-3D 127-class segmentation)."""

from __future__ import annotations

import modal

from modal_app.common import (
    gpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_premium_ct,
)

app = modal.App("manthana-premium-ct")


@app.function(
    image=service_image_premium_ct(),
    gpu="A10G",
    volumes={"/models": models_volume()},
    secrets=[manthana_secret()],
    **gpu_function_kwargs(timeout=900, scaledown_window=180, max_containers=2),
)
@modal.asgi_app()
def serve():
    import os
    import sys

    os.environ.setdefault("MANTHANA_LLM_REPO_ROOT", "/app")
    os.environ.setdefault("VISTA3D_ENABLED", "true")
    os.environ.setdefault(
        "VISTA3D_MODEL_PATH",
        "/models/vista3d/vista3d_pretrained_model/model.safetensors",
    )
    os.environ.setdefault("VISTA3D_FULL_FORWARD", "true")
    os.environ.setdefault("DEVICE", "cuda")

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    from main import app as fastapi_app

    return fastapi_app

