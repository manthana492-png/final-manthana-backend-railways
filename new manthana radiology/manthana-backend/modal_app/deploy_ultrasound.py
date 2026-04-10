"""Deploy ultrasound (Rad-DINO + narrative_usg) to Modal: FastAPI ASGI on GPU."""

from __future__ import annotations

import modal

from modal_app.common import (
    gpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_ultrasound,
)

app = modal.App("manthana-ultrasound")


@app.function(
    image=service_image_ultrasound(),
    gpu="T4",
    volumes={"/models": models_volume()},
    secrets=[manthana_secret()],
    **gpu_function_kwargs(timeout=600, scaledown_window=90, max_containers=4),
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
