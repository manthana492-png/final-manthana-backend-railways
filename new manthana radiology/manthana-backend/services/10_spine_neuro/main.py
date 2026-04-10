"""Manthana — Spine/Neuro Service: TotalSeg vertebrae"""
import json
import os
import shutil
import sys
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.normpath(os.path.join(_ROOT, "..", "..", "shared")),
    "/app/shared",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from config import PORT, SERVICE_NAME
from service_upload_prep import prepare_upload_for_pipeline

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


@app.get("/health")
async def health():
    import torch

    from inference import is_loaded

    ch = is_loaded()
    ok = ch.get("ready", False) if isinstance(ch, dict) else bool(ch)
    return {
        "service": SERVICE_NAME,
        "status": "ok" if ok else "degraded",
        "models_loaded": ok,
        "component_health": ch,
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/ready")
async def ready():
    from fastapi import HTTPException

    from inference import is_loaded

    ch = is_loaded()
    if not ch.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="Spine/Neuro service not ready (TotalSegmentator required).",
        )
    return {"ready": True, **ch}


@app.post("/analyze/spine_neuro")
async def analyze(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    series_dir: str = Form(""),
    patient_context_json: Optional[str] = Form(None),
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
    patient_context: dict = {}
    if patient_context_json:
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                patient_context = parsed
        except json.JSONDecodeError:
            patient_context = {}
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
            patient_context=patient_context,
        )
    except HTTPException:
        raise
    except Exception as e:
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
