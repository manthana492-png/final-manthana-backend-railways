"""
Manthana — ECG Service
AI-powered ECG analysis with camera photo support for rural doctors.

Pipeline:
  1. Input detection (photo vs signal data)
  2. If photo → digitizer or simplified CV extraction
  3. Heuristic rhythm ensemble (Manthana-ECG-Engine) + neurokit2 intervals
  4. Structured findings (List[Finding]) + interval dict in structures
"""

import json
import os
import sys
import time
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

sys.path.insert(0, "/app/shared")

from config import SERVICE_NAME, PORT
from preprocessing.ecg_utils import detect_input_type

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


def _parse_patient_context_json(raw: str) -> dict:
    if not raw or not str(raw).strip():
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


@app.get("/health")
async def health():
    from inference import is_loaded
    import torch
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "models_loaded": is_loaded(),
        "gpu_available": torch.cuda.is_available(),
        "version": "1.0.0",
    }


@app.post("/analyze/ecg")
async def analyze_ecg(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    patient_context_json: str = Form(""),
):
    """Analyze ECG — supports camera photos, CSV, EDF, DICOM-ECG."""
    from inference import run_ecg_pipeline
    from schemas import AnalysisResponse

    if job_id is None:
        job_id = str(uuid.uuid4())

    start = time.time()

    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".dat"
    filepath = os.path.join(upload_dir, f"{job_id}{ext}")

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    if detect_input_type(filepath) == "pdf_rejected":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unsupported_format",
                "message": (
                    "PDF ECG files cannot be processed directly. "
                    "Please upload the digital ECG file (.csv, .edf, .dcm) "
                    "or a clear photo (.jpg, .png) of the ECG printout."
                ),
            },
        )

    try:
        result = run_ecg_pipeline(
            filepath,
            job_id,
            patient_context=_parse_patient_context_json(patient_context_json),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ECG analysis failed: {str(e)}") from e

    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id

    return AnalysisResponse(**result).model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
