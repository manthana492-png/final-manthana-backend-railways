"""Deploy oral_cancer to Modal: **production-default** GPU (T4) ASGI.

**Gateway:** set ``ORAL_CANCER_SERVICE_URL`` to this app's HTTPS origin + ``/analyze/oral_cancer``
(Railway ``manthana-env``). Use ``deploy_oral_cancer_cpu.py`` only for cost-sensitive / low-volume
workloads without UNI; this GPU app is the right default for V2-M + B3 + optional **UNI** histopath.

**Secrets (``manthana-env``):** ``OPENROUTER_API_KEY``, ``ORAL_CANCER_ENABLED=true`` (default),
``HF_TOKEN`` if using UNI, ``MODEL_DIR=/models`` (volume mounted below).
"""

from __future__ import annotations

import modal

from modal_app.common import (
    gpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_oral_cancer,
)

app = modal.App("manthana-oral-cancer")


@app.function(
    image=service_image_oral_cancer(),
    gpu="T4",
    memory=8192,
    volumes={"/models": models_volume()},
    secrets=[manthana_secret()],
    **gpu_function_kwargs(timeout=600, scaledown_window=90, max_containers=4),
)
@modal.asgi_app()
def serve():
    import os
    import sys

    os.environ.setdefault("MANTHANA_LLM_REPO_ROOT", "/app")
    os.environ.setdefault("DEVICE", "cuda")
    os.environ.setdefault("MODEL_DIR", "/models")
    os.environ.setdefault("MANTHANA_MODEL_CACHE", "/models")
    os.environ.setdefault("ORAL_CANCER_ENABLED", "true")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    from main import app as fastapi_app

    return fastapi_app
