r"""Deploy ECG service to Modal: CPU-only FastAPI ASGI with scale-to-zero.

**No Modal volume** — ECG uses heuristics + neurokit2 + OpenRouter ``narrative_ecg`` only; no weight uploads.

Cost: ~$0 when idle (scales to zero), ~$0.0000131/core/sec when active.

Deploy:
  cd new\ manthana\ radiology/manthana-backend
  modal deploy modal_app/deploy_ecg.py

Railway: ECG_SERVICE_URL=https://<username>--manthana-ecg-<hash>.modal.run (path must include /analyze/ecg)
"""

from __future__ import annotations

import modal

from modal_app.common import (
    cpu_function_kwargs,
    manthana_secret,
    service_image_ecg_cpu,
)

app = modal.App("manthana-ecg")


@app.function(
    image=service_image_ecg_cpu(),
    cpu=2.0,  # 2 physical cores - plenty for ECG signal processing
    memory=4096,  # 4GB RAM for model loading
    secrets=[manthana_secret()],
    **cpu_function_kwargs(timeout=300, scaledown_window=60, max_containers=3),
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
