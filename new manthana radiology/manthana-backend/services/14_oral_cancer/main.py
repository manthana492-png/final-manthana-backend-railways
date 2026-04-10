"""
Manthana — Oral Cancer Screening Service
Clinical intraoral photos, phone images, and optional histopathology crops (UNI path).

Weights are optional: EfficientNet-V2-M (default before B3 when both exist), EfficientNet-B3 checkpoint, UNI + head;
OpenRouter (role oral_cancer, SSOT cloud_inference.yaml) for vision JSON and narrative when needed.
"""

import json
import logging
import os
import sys
import time
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (
    "/app/shared",
    os.path.normpath(os.path.join(_here, "..", "..", "shared")),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

from config import SERVICE_NAME, PORT
from inference import (
    OralServiceUnavailableError,
    get_loaded_status,
    is_service_ready,
    oral_degraded_from_exception,
    run_oral_cancer_pipeline,
)

logger = logging.getLogger("manthana.oral_cancer.main")

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


@app.get("/health")
async def health():
    status = get_loaded_status()
    gpu_ok = False
    try:
        import torch

        gpu_ok = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "models_loaded": status["ready"],
        "component_health": status,
        "ready": status["ready"],
        "gpu_available": gpu_ok,
        "version": "1.0.0",
    }


@app.get("/ready")
async def ready():
    """Readiness when service is enabled (local weights optional — degraded paths allowed)."""
    if not is_service_ready():
        raise HTTPException(
            status_code=503,
            detail="Oral cancer service disabled (ORAL_CANCER_ENABLED=false).",
        )
    return {"ready": True, "component_health": get_loaded_status()}


@app.post("/analyze/oral_cancer")
async def analyze_oral_cancer(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    clinical_notes: str = Form(""),
    patient_context_json: str = Form(""),
    input_type: str = Form(""),
):
    """Screen oral cavity photo or histopathology crop; optional patient context JSON and input_type hint."""
    from schemas import AnalysisResponse

    if job_id is None:
        job_id = str(uuid.uuid4())

    if not is_service_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "oral_cancer_service_disabled",
                "message": "Oral cancer service is disabled (ORAL_CANCER_ENABLED=false).",
            },
        )

    patient_context: dict = {}
    if patient_context_json and str(patient_context_json).strip():
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                patient_context = parsed
        except json.JSONDecodeError:
            logger.warning("Invalid patient_context_json; ignoring.")

    input_override = (input_type or "").strip() or None
    if input_override and input_override not in (
        "clinical_photo",
        "histopathology",
        "mixed",
        "unknown",
    ):
        input_override = None

    start = time.time()

    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    filepath = os.path.join(upload_dir, f"{job_id}{ext}")

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = run_oral_cancer_pipeline(
            filepath,
            job_id,
            clinical_notes=clinical_notes,
            patient_context=patient_context,
            input_type_override=input_override,
        )
    except OralServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Oral pipeline unexpected error; returning degraded response: %s", e)
        result = oral_degraded_from_exception(job_id, clinical_notes, patient_context, str(e))

    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id
    if patient_id:
        result.setdefault("structured", {})
        if isinstance(result["structured"], dict):
            result["structured"]["patient_id"] = patient_id

    return AnalysisResponse(**result).model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
