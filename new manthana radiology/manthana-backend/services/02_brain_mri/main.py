"""
Manthana — Brain MRI Service
TotalSegmentator total_mr + SynthSeg + optional Prima pipeline.
"""
import json
import os
import sys
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

_MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB_BRAIN_MRI", "512") or "512")
_MAX_UPLOAD_BYTES = int(_MAX_UPLOAD_MB * 1024 * 1024)


def _allowed_brain_mri_filename(name: str) -> bool:
    l = (name or "").lower()
    if l.endswith(".nii.gz"):
        return True
    return l.endswith((".nii", ".dcm", ".dic", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))

_BACKEND_SHARED = os.path.normpath(os.path.join(_MAIN_DIR, "..", "..", "shared"))
for _p in (_BACKEND_SHARED, "/app/shared"):
    if os.path.isdir(_p):
        while _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
from config import PORT, SERVICE_NAME

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


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
                    "Allowed uploads: NIfTI (.nii, .nii.gz), DICOM (.dcm/.dic), "
                    "or raster (.png/.jpg/.jpeg/.webp/.bmp/.gif) for degraded analysis."
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
    try:
        result = run_brain_mri_pipeline(
            filepath,
            job_id,
            series_dir=series_dir or "",
            clinical_notes=merged_notes,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "inference_failed",
                "message": str(e),
            },
        ) from e
    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id
    return AnalysisResponse(**result).model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
