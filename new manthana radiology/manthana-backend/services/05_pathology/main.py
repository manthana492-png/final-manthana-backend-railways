"""Manthana — Pathology Service: Virchow tile embeddings"""
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ── Shared path resolver ────────────────────────────────


def _find_shared() -> Path:
    """Resolve the shared/ directory whether running Docker, Lightning AI, or locally."""
    candidates = [
        Path("/app/shared"),
        Path(__file__).resolve().parent.parent.parent / "shared",
        Path(__file__).resolve().parent.parent / "shared",
        Path.cwd() / "shared",
        Path.cwd().parent / "shared",
    ]
    for c in candidates:
        if (c / "schemas.py").exists():
            return c
    raise RuntimeError(
        "Cannot find shared/ directory. Searched: " + ", ".join(str(c) for c in candidates)
    )


_shared = _find_shared()
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from config import PORT, SERVICE_NAME

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

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


@app.post("/analyze/pathology")
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
    ext = os.path.splitext(file.filename or "")[1] or ".svs"
    fp = os.path.join(upload_dir, f"{job_id}{ext}")
    with open(fp, "wb") as f:
        f.write(await file.read())

    patient_context: dict = {}
    if patient_context_json:
        try:
            patient_context = json.loads(patient_context_json)
        except json.JSONDecodeError:
            patient_context = {}

    try:
        result = run_pipeline(
            fp,
            job_id,
            patient_context=patient_context,
            claude_client=None,
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
