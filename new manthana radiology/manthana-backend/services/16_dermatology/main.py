"""
Manthana — Dermatology service (Kimi K2.5 vision + DermAI system prompt + optional V2 weights).
"""

from __future__ import annotations

import base64
import json
import sys
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, "/app/shared")

from config import DERM_MAX_UPLOAD_BYTES, PORT, SERVICE_NAME
from analyzer import analyze_dermatology, get_ready

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/ready")
async def ready():
    """Readiness: requires Kimi (Moonshot) API key."""
    st = get_ready()
    if not st.get("ready"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Service not ready: set KIMI_API_KEY or MOONSHOT_API_KEY, "
                "and ensure prompts/dermatology_dermai_system.txt is present in the image."
            ),
        )
    return st


@app.post("/analyze/dermatology")
async def analyze_dermatology_endpoint(
    file: UploadFile = File(...),
    job_id: str = Form(""),
    clinical_notes: str = Form(""),
    patient_id: str = Form(""),
    patient_context_json: str | None = Form(None),
):
    from schemas import AnalysisResponse

    if not job_id:
        job_id = str(uuid.uuid4())

    content = await file.read()
    if len(content) > DERM_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: max {DERM_MAX_UPLOAD_BYTES} bytes",
        )
    image_b64 = base64.b64encode(content).decode()

    patient_context: dict = {}
    if patient_context_json:
        try:
            patient_context = json.loads(patient_context_json)
            if not isinstance(patient_context, dict):
                patient_context = {}
        except json.JSONDecodeError:
            patient_context = {}
    if clinical_notes:
        patient_context["clinical_notes"] = clinical_notes
    if patient_id:
        patient_context.setdefault("patient_id", patient_id)

    start = time.time()
    try:
        result = analyze_dermatology(
            image_b64=image_b64,
            patient_context=patient_context,
            job_id=job_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id
    return AnalysisResponse(**result).model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
