"""Deploy Brain MRI to Modal: FastAPI ASGI on GPU."""

from __future__ import annotations

import modal

from modal_app.common import (
    gpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_brain_mri,
)

app = modal.App("manthana-brain-mri")


@app.function(
    image=service_image_brain_mri(),
    gpu="L4",
    volumes={"/models": models_volume()},
    secrets=[manthana_secret()],
    **gpu_function_kwargs(timeout=900, scaledown_window=90, max_containers=3),
)
@modal.asgi_app()
def serve():
    import os
    import sys

    os.environ.setdefault("MANTHANA_LLM_REPO_ROOT", "/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    from main import app as fastapi_app

    return fastapi_app
