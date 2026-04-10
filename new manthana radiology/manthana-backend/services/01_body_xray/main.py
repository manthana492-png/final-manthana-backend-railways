"""
Manthana — Universal Body X-Ray Service
User uploads ANY X-ray → auto-detects body region → routes to the appropriate pipeline.

Pipelines:
  Chest → TorchXRayVision dual-model ensemble (all + chex or mimic_nb fallback)
  Bone / extremity / spine / skull → deterministic OpenCV/numeric scoring (pipeline_bone)
  Abdomen / pelvis → deterministic OpenCV/numeric scoring (pipeline_abdomen)
"""

import json
import os
import sys
import time
import uuid
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

sys.path.insert(0, "/app/shared")

from image_intake import (
    intake_pil_to_temp_path,
    merge_image_quality_into_result,
    normalize_for_model,
)

from config import SERVICE_NAME, PORT

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


@app.get("/health")
async def health():
    import torch
    from pipeline_chest import is_loaded as chest_loaded
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "models_loaded": chest_loaded(),
        "gpu_available": torch.cuda.is_available(),
        "version": "1.0.0",
    }


@app.post("/analyze/xray")
async def analyze_xray(
    file: UploadFile = File(...),
    job_id: str = Form(None),
    patient_id: str = Form(""),
    patient_context_json: Optional[str] = Form(None),
):
    """Analyze any body X-ray — auto-detects region."""
    from body_detector import detect_body_region
    from schemas import AnalysisResponse

    if job_id is None:
        job_id = str(uuid.uuid4())

    patient_context = None
    if patient_context_json:
        try:
            patient_context = json.loads(patient_context_json)
        except json.JSONDecodeError:
            patient_context = None
    
    start = time.time()

    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)

    raw = await file.read()
    try:
        intake = normalize_for_model(raw, modality="xray")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    filepath = intake_pil_to_temp_path(
        intake["pil"], upload_dir, job_id, prefer_grayscale=True
    )
    image_b64 = base64.b64encode(raw).decode("ascii")
    try:
        # Auto-detect body region
        region = detect_body_region(filepath)

        # Route to correct pipeline
        if region in ("chest", "thorax"):
            from inference import run_pipeline

            result = run_pipeline(filepath, job_id, patient_context=patient_context)
        elif region in ("extremity", "hand", "wrist", "knee", "ankle", "elbow", "shoulder", "hip"):
            from pipeline_bone import run_bone_pipeline
            from inference import attach_narrative

            result = run_bone_pipeline(filepath, job_id, region)
            result = attach_narrative(
                result,
                patient_context=patient_context,
                image_b64=image_b64,
            )
        elif region in ("spine", "cervical", "thoracic", "lumbar"):
            from pipeline_bone import run_spine_pipeline
            from inference import attach_narrative

            result = run_spine_pipeline(filepath, job_id, region)
            result = attach_narrative(
                result,
                patient_context=patient_context,
                image_b64=image_b64,
            )
        elif region in ("abdomen", "pelvis"):
            from pipeline_abdomen import run_abdomen_pipeline
            from inference import attach_narrative

            result = run_abdomen_pipeline(filepath, job_id, region)
            result = attach_narrative(
                result,
                patient_context=patient_context,
                image_b64=image_b64,
            )
        elif region in ("skull", "head"):
            from pipeline_bone import run_skull_pipeline
            from inference import attach_narrative

            result = run_skull_pipeline(filepath, job_id)
            result = attach_narrative(
                result,
                patient_context=patient_context,
                image_b64=image_b64,
            )
        else:
            # Default to chest (most common)
            from inference import run_pipeline

            result = run_pipeline(filepath, job_id, patient_context=patient_context)
            result["detected_region"] = f"{region} (defaulted to chest analysis)"

        result["detected_region"] = result.get("detected_region", region)
        result["processing_time_sec"] = round(time.time() - start, 2)
        result["job_id"] = job_id
        merge_image_quality_into_result(result, intake["quality"])

        return AnalysisResponse(**result).model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    finally:
        try:
            os.unlink(filepath)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
