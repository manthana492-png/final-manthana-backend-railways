"""Deploy Dermatology service to Modal: CPU-only FastAPI ASGI with scale-to-zero.

Uses OpenRouter vision API + optional local DermAI V2 weights (CPU-compatible).
Cost: ~$0 when idle (scales to zero), ~$0.0000131/core/sec when active.
Much cheaper than Railway always-on for bursty dermatology workloads.

Deploy:
  cd new\ manthana\ radiology/manthana-backend
  modal deploy modal_app/deploy_dermatology.py

Service URL will be printed after deploy - paste into:
  Railway Gateway env: DERMATOLOGY_SERVICE_URL=https://<username>--manthana-dermatology-[hash].modal.run
"""

from __future__ import annotations

import modal

from modal_app.common import (
    cpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_dermatology_cpu,
)

app = modal.App("manthana-dermatology")


@app.function(
    image=service_image_dermatology_cpu(),
    cpu=2.0,  # 2 physical cores - enough for image preprocessing + OpenRouter calls
    memory=6144,  # V2-M + optional Grad-CAM; reduce to 4096 if memory is tight and CAM is off
    volumes={"/models": models_volume()},  # Optional DermAI V2 weights
    secrets=[manthana_secret()],
    **cpu_function_kwargs(timeout=300, scaledown_window=60, max_containers=1),
)
@modal.asgi_app()
def serve():
    import os
    import sys

    os.environ.setdefault("MANTHANA_LLM_REPO_ROOT", "/app")
    os.environ.setdefault("DEVICE", "cpu")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    from main import app as fastapi_app

    return fastapi_app
