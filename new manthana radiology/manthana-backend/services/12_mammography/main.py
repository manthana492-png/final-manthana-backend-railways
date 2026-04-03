"""Manthana — Mammography Service: Mirai (4-view) + optional Claude narrative"""
import json
import os
import sys
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

sys.path.insert(0, "/app/shared")
from config import PORT, SERVICE_NAME

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


@app.post("/analyze/mammography")
async def analyze(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    patient_context_json: Optional[str] = Form(None),
):
    from inference import run_pipeline
    from schemas import AnalysisResponse

    if not job_id:
        job_id = str(uuid.uuid4())
    start = time.time()
    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    fp = os.path.join(upload_dir, f"{job_id}{os.path.splitext(file.filename or '')[1] or '.dcm'}")
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

    claude_client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    try:
        result = run_pipeline(
            filepath=fp,
            job_id=job_id,
            patient_context=patient_context,
            claude_client=claude_client,
            image_b64="",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    result["processing_time_sec"] = round(time.time() - start, 2)
    result["job_id"] = job_id
    return AnalysisResponse(**result).model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
