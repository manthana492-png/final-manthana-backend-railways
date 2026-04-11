"""Manthana — Premium unified 3D CT service (full VISTA-3D path)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

try:
    from manthana_paths import backend_root_from_service_file
except ImportError:

    def backend_root_from_service_file(p: Path | str) -> Path:
        x = Path(p).resolve()
        if os.environ.get("MANTHANA_BACKEND_ROOT"):
            return Path(os.environ["MANTHANA_BACKEND_ROOT"])
        if Path("/app/shared").is_dir() and x.parent == Path("/app"):
            return Path("/app")
        return x.parents[2]


_root = backend_root_from_service_file(__file__)
for _shared in (Path("/app/shared"), _root / "shared"):
    if _shared.is_dir():
        sys.path.insert(0, str(_shared))
        break

from config import PORT, SERVICE_NAME
from service_upload_prep import prepare_upload_for_pipeline

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


@app.get("/health")
async def health():
    import torch

    from inference import is_loaded

    ch = is_loaded()
    ready_inf = bool(ch.get("vista3d_enabled") and ch.get("vista3d_checkpoint_present"))
    return {
        "service": SERVICE_NAME,
        "status": "ok" if ready_inf else "degraded",
        "models_loaded": ready_inf,
        "component_health": ch,
        "gpu_available": torch.cuda.is_available(),
        "version": "1.0.0",
    }


@app.get("/ready")
async def ready():
    from inference import is_loaded

    ch = is_loaded()
    if ch.get("vista3d_enabled") and ch.get("vista3d_checkpoint_present"):
        return {"ready": True, **ch}
    raise HTTPException(
        status_code=503,
        detail="Premium CT service not ready: VISTA3D model checkpoint missing.",
    )


@app.post("/analyze/premium_ct")
async def analyze(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    series_dir: str = Form(""),
    source_modality: str = Form(""),
    patient_context_json: str = Form(""),
):
    from inference import run_pipeline
    from schemas import AnalysisResponse

    if not job_id:
        job_id = str(uuid.uuid4())
    start = time.time()
    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".dcm"
    fp = os.path.join(upload_dir, f"{job_id}{ext}")
    with open(fp, "wb") as f:
        f.write(await file.read())

    patient_ctx: dict = {}
    if patient_context_json and patient_context_json.strip():
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                patient_ctx = parsed
        except json.JSONDecodeError:
            patient_ctx = {}

    cleanup_dirs: list[str] = []
    try:
        if series_dir and os.path.isdir(series_dir):
            pipeline_path = fp
        else:
            pipeline_path, cleanup_dirs = prepare_upload_for_pipeline(fp, file.filename, job_id)
        result = run_pipeline(
            pipeline_path,
            job_id,
            series_dir=series_dir or "",
            patient_context=patient_ctx or None,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        for d in cleanup_dirs:
            shutil.rmtree(d, ignore_errors=True)
    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id
    return AnalysisResponse(**result).model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)

