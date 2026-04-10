"""
Manthana — Brain MRI Service
TotalSegmentator total_mr + SynthSeg + optional Prima pipeline.
"""
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

_MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB_BRAIN_MRI", "512") or "512")
_MAX_UPLOAD_BYTES = int(_MAX_UPLOAD_MB * 1024 * 1024)

_root = Path(__file__).resolve().parents[2]
for _shared in (_root / "shared", Path("/app/shared")):
    if _shared.is_dir():
        p = str(_shared)
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
        break

from config import PORT, SERVICE_NAME
from service_upload_prep import prepare_upload_for_pipeline

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


def _allowed_brain_mri_filename(name: str) -> bool:
    l = (name or "").lower()
    if l.endswith(".nii.gz"):
        return True
    return l.endswith(
        (
            ".nii",
            ".dcm",
            ".dic",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".gif",
            ".zip",
            ".heic",
            ".heif",
        )
    )


@app.get("/health")
async def health():
    import torch

    from inference import is_loaded

    ch = is_loaded()
    return {
        "service": SERVICE_NAME,
        "status": "ok" if ch.get("ready") else "degraded",
        "models_loaded": ch,
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
            detail="Brain MRI service not ready (TotalSegmentator required).",
        )
    return {"ready": True, **ch}


@app.post("/analyze/brain_mri")
async def analyze(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    series_dir: str = Form(""),
    clinical_notes: str = Form(""),
    patient_context_json: str = Form(""),
):
    from inference import run_brain_mri_pipeline
    from schemas import AnalysisResponse

    if job_id is None:
        job_id = str(uuid.uuid4())
    if not _allowed_brain_mri_filename(file.filename or ""):
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "invalid_file_type",
                "message": (
                    "Allowed uploads: NIfTI (.nii, .nii.gz), DICOM (.dcm/.dic), ZIP of images, "
                    "or raster (.png/.jpg/.jpeg/.webp/.bmp/.gif/.heic) for degraded or film-photo analysis."
                ),
            },
        )
    merged_notes = (clinical_notes or "").strip()
    pc = (patient_context_json or "").strip()
    if pc:
        try:
            parsed = json.loads(pc)
            extra = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else pc
        except json.JSONDecodeError:
            extra = pc
        merged_notes = f"{merged_notes}\n{extra}".strip() if merged_notes else extra
    start = time.time()
    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error_code": "payload_too_large",
                "message": f"Upload exceeds MAX_UPLOAD_MB_BRAIN_MRI limit ({int(_MAX_UPLOAD_MB)} MiB).",
            },
        )
    fn = (file.filename or "").lower()
    if fn.endswith(".nii.gz"):
        ext = ".nii.gz"
    else:
        ext = os.path.splitext(file.filename or "")[1] or ".dcm"
    filepath = os.path.join(upload_dir, f"{job_id}{ext}")
    with open(filepath, "wb") as f:
        f.write(raw)
    cleanup_dirs: list[str] = []
    try:
        if series_dir and os.path.isdir(series_dir):
            pipeline_path = filepath
        else:
            pipeline_path, cleanup_dirs = prepare_upload_for_pipeline(filepath, file.filename, job_id)
        result = run_brain_mri_pipeline(
            pipeline_path,
            job_id,
            series_dir=series_dir or "",
            clinical_notes=merged_notes,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "inference_failed",
                "message": str(e),
            },
        ) from e
    finally:
        for d in cleanup_dirs:
            shutil.rmtree(d, ignore_errors=True)
    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id
    return AnalysisResponse(**result).model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
