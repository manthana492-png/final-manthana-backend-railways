"""Deploy CT Brain VISTA-3D premium tier to Modal (A10G, shared volume)."""

from __future__ import annotations

import modal

from modal_app.common import (
    gpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_ct_brain_vista,
)

app = modal.App("manthana-ct-brain-vista")


@app.function(
    image=service_image_ct_brain_vista(),
    gpu="A10G",
    volumes={"/models": models_volume()},
    secrets=[manthana_secret()],
    **gpu_function_kwargs(timeout=600, scaledown_window=120, max_containers=2),
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
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    from main import app as fastapi_app

    return fastapi_app
