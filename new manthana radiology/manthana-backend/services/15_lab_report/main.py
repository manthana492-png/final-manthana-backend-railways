"""
Manthana — Lab Report Analysis Service
Interprets PDF/text lab reports using OpenRouter + optional Parrotlet VLM.

Input: multipart file or JSON { document_b64, patient_context_json }
Output: Structured findings with E2E enrichment (test_results, patterns, narrative).
"""

import json
import logging
import os
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException, Request

sys.path.insert(0, "/app/shared")

from config import SERVICE_NAME, PORT

logger = logging.getLogger("manthana.lab_report")

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


def _finalize_lab_api_response(result: dict, job_id: str, start: float) -> dict:
    """Merge job metadata, Option A critical findings, AnalysisResponse."""
    from schemas import AnalysisResponse

    result = dict(result)
    result["job_id"] = job_id
    result["modality"] = "lab_report"
    result["status"] = "complete"
    result["processing_time_sec"] = round(time.time() - start, 2)
    result.setdefault("confidence", "medium")

    findings = result.get("findings") or []
    if not isinstance(findings, list):
        findings = []
    for alert in reversed(result.get("critical_values") or []):
        if isinstance(alert, str):
            findings.insert(
                0,
                {
                    "label": "CRITICAL VALUE",
                    "description": alert,
                    "severity": "critical",
                    "confidence": 100.0,
                    "region": "Laboratory",
                },
            )
    result["findings"] = findings

    return AnalysisResponse(**result).model_dump()


@app.get("/health")
async def health():
    or_ok = False
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"):
        k = (os.environ.get(name) or "").strip()
        if k and len(k) >= 8:
            or_ok = True
            break
    return {
        "service": SERVICE_NAME,
        "status": "ok" if or_ok else "no_api_key",
        "models_loaded": or_ok,
        "llm": "openrouter",
        "gpu_available": False,
        "version": "1.0.0",
    }


@app.get("/ready")
async def ready():
    from analyzer import is_service_ready

    info = is_service_ready()
    if not info.get("ready"):
        raise HTTPException(status_code=503, detail=info)
    return info


@app.post("/analyze/lab_report")
async def analyze_lab_report_endpoint(request: Request):
    """
    Multipart: file, optional job_id, patient_id, clinical_notes.
    JSON: document_b64, patient_context_json (string or object), optional job_id, filename_hint.
    """
    from analyzer import analyze_lab_report, parse_clinical_notes
    from inference import enrich_lab_pipeline_output, run_lab_report_pipeline_b64

    job_id = str(uuid.uuid4())
    start = time.time()
    ct = (request.headers.get("content-type") or "").lower()

    try:
        if "application/json" in ct:
            body = await request.json()
            job_id = body.get("job_id") or job_id
            pctx = body.get("patient_context_json", "{}")
            if not isinstance(pctx, str):
                pctx = json.dumps(pctx)
            result = run_lab_report_pipeline_b64(
                body.get("document_b64") or "",
                patient_context_json=pctx,
                filename_hint=body.get("filename_hint") or "report.pdf",
            )
            if result.get("available") is False:
                raise HTTPException(
                    status_code=400,
                    detail=result.get("message") or result.get("reason") or "invalid request",
                )
        else:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(status_code=422, detail="multipart field 'file' is required")
            job_id = form.get("job_id") or job_id
            patient_id = form.get("patient_id") or ""
            clinical_notes = form.get("clinical_notes") or ""

            upload_dir = "/tmp/manthana_uploads"
            os.makedirs(upload_dir, exist_ok=True)
            fname = getattr(upload, "filename", None) or "report.pdf"
            ext = os.path.splitext(fname)[1] or ".pdf"
            filepath = os.path.join(upload_dir, f"{job_id}{ext}")

            raw_bytes = await upload.read()
            with open(filepath, "wb") as f:
                f.write(raw_bytes)

            patient_context = parse_clinical_notes(str(clinical_notes))
            if patient_id:
                patient_context = {**patient_context, "patient_id": str(patient_id)}

            try:
                result = analyze_lab_report(filepath, patient_context=patient_context)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                logger.exception("Lab analysis failed: %s", e)
                raise HTTPException(status_code=500, detail=f"Lab analysis failed: {str(e)}") from e

            result = enrich_lab_pipeline_output(result, patient_context)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Lab endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return _finalize_lab_api_response(result, job_id, start)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
