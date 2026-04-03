"""Manthana — Abdominal CT Service: TotalSeg + Comp2Comp CLI (series)"""
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
import zipfile

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

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


def _prepare_upload_for_pipeline(saved_path: str, filename: str | None, job_id: str) -> tuple[str, list[str]]:
    """
    Return (path_for_run_pipeline, cleanup_dir_paths).

    - ZIP → extract to temp dir (DICOM series).
    - Single .dcm → temp dir with one instance (TotalSegmentator expects a folder).
    - Raster / NIfTI → use saved file path as-is.
    """
    cleanup: list[str] = []
    ext = (os.path.splitext(filename or "")[1] or "").lower()
    lower_name = (filename or "").lower()

    if ext == ".zip" or lower_name.endswith(".zip"):
        d = tempfile.mkdtemp(prefix=f"ct_zip_{job_id}_")
        cleanup.append(d)
        try:
            with zipfile.ZipFile(saved_path, "r") as zf:
                zf.extractall(d)
        except zipfile.BadZipFile as e:
            shutil.rmtree(d, ignore_errors=True)
            cleanup.clear()
            raise HTTPException(status_code=400, detail=f"Invalid ZIP: {e}") from e
        return d, cleanup

    if ext in (".dcm", ".dic") or lower_name.endswith((".dcm", ".dic")):
        d = tempfile.mkdtemp(prefix=f"ct_series_{job_id}_")
        cleanup.append(d)
        dest_name = os.path.basename(filename) if filename else f"instance{ext or '.dcm'}"
        if not dest_name.lower().endswith((".dcm", ".dic")):
            dest_name = "instance.dcm"
        shutil.copy2(saved_path, os.path.join(d, dest_name))
        return d, cleanup

    return saved_path, cleanup


def _count_dicoms_in_tree(root: str) -> int:
    n = 0
    for _, _, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if fl.endswith(".dcm") or fl.endswith(".dic"):
                n += 1
    return n


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
            pipeline_path, cleanup_dirs = _prepare_upload_for_pipeline(fp, file.filename, job_id)

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
