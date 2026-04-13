"""
Modal: CXR MedGemma + Kimi orchestration (workspace 2 pattern).

Deploy from ``manthana-backend`` root (same as other GPU apps)::

    modal deploy modal_app/deploy_cxr_medgemma.py

Expects Modal secret ``manthana-env`` with OPENROUTER_API_KEY and HF token if needed.
TorchXRayVision scores + image come from ``manthana-body-xray`` with ``skip_llm_narrative=true``.
"""

from __future__ import annotations

import modal

from modal_app.common import (
    gpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_cxr_medgemma,
)

app = modal.App("manthana-cxr-medgemma")


@app.function(
    image=service_image_cxr_medgemma(),
    gpu="A10G",
    volumes={"/models": models_volume()},
    secrets=[manthana_secret()],
    **gpu_function_kwargs(timeout=900, scaledown_window=120, max_containers=1),
)
@modal.asgi_app()
def serve():
    import os
    import sys

    os.environ.setdefault("MANTHANA_LLM_REPO_ROOT", "/app")
    os.environ.setdefault("DEVICE", "cuda")
    os.environ.setdefault("PORT", "8019")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    from main import app as fastapi_app

    return fastapi_app
