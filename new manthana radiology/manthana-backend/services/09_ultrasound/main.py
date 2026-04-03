"""Manthana — Ultrasound Service: OpenUS + MedSAM2"""

import os
import sys
import time
import uuid
from typing import Optional

# Ensure backend root and shared are on path for local/dev and uvicorn module runs.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED = os.path.join(_BACKEND_ROOT, "shared")
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from config import SERVICE_NAME, PORT
from schemas import AnalysisResponse, Finding
from disclaimer import DISCLAIMER

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


@app.get("/health")
async def health():
    from inference import is_loaded
    import torch

    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "models_loaded": is_loaded(),
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/analyze/ultrasound")
async def analyze(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    patient_context_json: Optional[str] = Form(None),
):
    import json
    import inference

    if not job_id:
        job_id = str(uuid.uuid4())
    start = time.time()
    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    fp = os.path.join(upload_dir, f"{job_id}{ext}")
    with open(fp, "wb") as f:
        f.write(await file.read())
    patient_context = {}
    if patient_context_json:
        try:
            patient_context = json.loads(patient_context_json)
        except Exception:
            patient_context = {}
    try:
        result = inference.run_pipeline(fp, job_id, patient_context=patient_context)
        result = inference.enrich_usg_pipeline_output(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id
    # Map legacy string narrative into structured findings list for AnalysisResponse
    findings_text = result.get("findings", "")
    structured = [
        Finding(
            label="Ultrasound summary",
            severity="info",
            confidence=0.0,
            description=findings_text,
        )
    ] if isinstance(findings_text, str) else findings_text
    result["findings"] = structured
    return AnalysisResponse(**result).model_dump()


class USGAnalysisRequest(BaseModel):
    image_b64: str
    patient_context_json: Optional[str] = None
    job_id: Optional[str] = None
    filename_hint: Optional[str] = None

    class Config:
        extra = "ignore"


@app.post("/analyze/ultrasound/json", response_model=AnalysisResponse)
async def analyze_usg_json(req: USGAnalysisRequest):
    """
    JSON body alternative to multipart.  Used by ZeroClaw agent and gateway JSON path.
    Content-Type: application/json
    """
    from inference import run_usg_pipeline_b64, enrich_usg_pipeline_output

    result = run_usg_pipeline_b64(
        image_b64=req.image_b64,
        patient_context_json=req.patient_context_json,
        job_id=req.job_id,
    )
    if not result.get("available", True):
        return AnalysisResponse(
            modality="ultrasound",
            findings=result.get("findings", "Input error."),
            impression="Analysis unavailable.",
            pathology_scores={
                "available": False,
                "reason": result.get("reason", "unknown"),
            },
            structures=[],
            confidence="low",
            models_used=["Manthana Ultrasound Engine"],
            disclaimer=DISCLAIMER,
            job_id=req.job_id or "none",
        )
    result = enrich_usg_pipeline_output(result)
    return AnalysisResponse(**result, job_id=req.job_id or "none")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
