"""Manthana — Abdominal CT Service: TotalSeg + Comp2Comp CLI (series)"""
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid

from dicom_utils_helpers import count_dicoms_in_tree as _count_dicoms_in_tree
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

logger = logging.getLogger("manthana.abdominal_ct.main")

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.normpath(os.path.join(_ROOT, "..", "..", "shared")),
    "/app/shared",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from config import PORT, SERVICE_NAME
from service_upload_prep import prepare_upload_for_pipeline as _prepare_upload_for_pipeline

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


@app.get("/health")
async def health():
    import torch

    from inference import is_loaded

    ch = is_loaded()
    ok = all(ch.values()) if isinstance(ch, dict) else bool(ch)
    return {
        "service": SERVICE_NAME,
        "status": "ok" if ok else "degraded",
        "models_loaded": ok,
        "component_health": ch,
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/analyze/abdominal_ct")
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
    patient_ctx: dict = {}
    if patient_context_json and patient_context_json.strip():
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                patient_ctx = parsed
        except json.JSONDecodeError:
            patient_ctx = {}
    start = time.time()
    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".dcm"
    fp = os.path.join(upload_dir, f"{job_id}{ext}")
    with open(fp, "wb") as f:
        f.write(await file.read())

    cleanup_dirs: list[str] = []
    try:
        if series_dir and os.path.isdir(series_dir):
            pipeline_path = fp
        else:
            pipeline_path, cleanup_dirs = _prepare_upload_for_pipeline(
                fp, file.filename, job_id
            )

        n_dcm = _count_dicoms_in_tree(pipeline_path) if os.path.isdir(pipeline_path) else 0
        declared = 0
        try:
            declared = int(patient_ctx.get("declared_file_count") or 0)
        except (TypeError, ValueError):
            declared = 0
        dicom_mismatch = bool(declared > 0 and n_dcm > 0 and n_dcm < declared * 0.5)
        if dicom_mismatch:
            logger.warning(
                "ZIP/folder has %s .dcm files; client declared %s — continuing with warning flag",
                n_dcm,
                declared,
            )

        result = run_pipeline(
            pipeline_path,
            job_id,
            series_dir=series_dir or "",
            source_modality=source_modality or "",
            patient_context=patient_ctx or None,
            http_upload_filename=file.filename or "",
            dicom_slices_found=n_dcm,
            dicom_declared_mismatch=dicom_mismatch,
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
