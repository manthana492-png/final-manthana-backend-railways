"""Deploy Oral Cancer to Modal: **CPU-only** alternate (cost / low volume).

**Production default:** `deploy_oral_cancer.py` (GPU T4) + `ORAL_CANCER_SERVICE_URL` — use this CPU
app only if you accept slower runs and limited UNI histopath UX.

Uses EfficientNet-V2-M and/or B3 on CPU + OpenRouter fallback; scale-to-zero.

Deploy:
  cd new\ manthana\ radiology/manthana-backend
  modal deploy modal_app/deploy_oral_cancer_cpu.py

If replacing the GPU app in Railway: ORAL_CANCER_SERVICE_URL=https://<username>--manthana-oral-cancer-cpu-[hash].modal.run/analyze/oral_cancer
"""

from __future__ import annotations

import modal

from modal_app.common import (
    cpu_function_kwargs,
    manthana_secret,
    models_volume,
    service_image_oral_cancer_cpu,
)

app = modal.App("manthana-oral-cancer-cpu")


@app.function(
    image=service_image_oral_cancer_cpu(),
    cpu=2.0,  # 2 physical cores - EfficientNet-B3 runs reasonably on CPU
    memory=4096,  # 4GB RAM for image processing
    volumes={"/models": models_volume()},  # oral_effnet_v2m.pt, B3 checkpoint, optional UNI head
    secrets=[manthana_secret()],
    **cpu_function_kwargs(timeout=300, scaledown_window=60, max_containers=3),
)
@modal.asgi_app()
def serve():
    import os
    import sys

    os.environ.setdefault("MANTHANA_LLM_REPO_ROOT", "/app")
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("MODEL_DIR", "/models")
    os.environ.setdefault("MANTHANA_MODEL_CACHE", "/models")
    os.environ.setdefault("ORAL_CANCER_ENABLED", "true")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/shared")
    from main import app as fastapi_app

    return fastapi_app
