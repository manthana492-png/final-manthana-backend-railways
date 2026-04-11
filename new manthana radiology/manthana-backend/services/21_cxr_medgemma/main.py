"""
Manthana — CXR MedGemma middle layer (workspace 2).

Consumes TorchXRayVision scores + same chest image + patient context from the universal
X-ray service (optionally with ``skip_llm_narrative=true``), runs MedGemma for structured
draft + follow-up questions, then Kimi (OpenRouter) for the final narrative after answers.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

sys.path.insert(0, "/app/shared")

from config import (  # noqa: E402
    KIMI_REPORT_ROLE,
    MAX_QUESTIONS,
    MAX_QUESTIONS_PER_SESSION,
    MIN_QUESTIONS,
    PORT,
    SERVICE_NAME,
    SESSION_TIMEOUT_MINUTES,
)

from medgemma_cxr_pipeline import (  # noqa: E402
    build_kimi_user_payload,
    normalize_answers_from_payload,
    run_medgemma_stage1,
)
from session_store import SessionStore  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manthana.cxr_medgemma")

SESSION_DIR = Path(os.getenv("CXR_MEDGEMMA_SESSION_DIR", "/tmp/manthana_cxr_medgemma_sessions"))
SESSION_DIR.mkdir(parents=True, exist_ok=True)

_sessions = SessionStore(ttl_seconds=float(SESSION_TIMEOUT_MINUTES * 60))

app = FastAPI(title=f"Manthana — {SERVICE_NAME}", version="1.0.0")


@app.get("/health")
async def health():
    try:
        from medical_document_parser import is_loaded  # type: ignore
    except Exception:
        is_loaded = lambda: False  # noqa: E731
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "medgemma_loaded": bool(is_loaded()),
        "port": PORT,
    }


def _clamp_questions(stage1: Dict[str, Any]) -> None:
    qs = stage1.get("follow_up_questions")
    if not isinstance(qs, list):
        stage1["follow_up_questions"] = []
        return
    cap = min(MAX_QUESTIONS, MAX_QUESTIONS_PER_SESSION)
    stage1["follow_up_questions"] = qs[:cap]
    if len(stage1["follow_up_questions"]) < MIN_QUESTIONS:
        logger.warning(
            "MedGemma returned fewer than MIN_QUESTIONS=%s (got %s)",
            MIN_QUESTIONS,
            len(stage1["follow_up_questions"]),
        )


def _jpeg_b64_from_path(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


class CompleteBody(BaseModel):
    session_id: str = Field(..., min_length=8)
    answers: Optional[Dict[str, str]] = None
    skip_all: bool = False


@app.post("/medgemma-cxr/session/start")
async def session_start(
    file: UploadFile = File(...),
    pathology_scores_json: str = Form(...),
    patient_context_json: Optional[str] = Form(None),
):
    try:
        scores = json.loads(pathology_scores_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"invalid pathology_scores_json: {e}") from e
    if not isinstance(scores, dict):
        raise HTTPException(status_code=422, detail="pathology_scores_json must be a JSON object")

    ctx: Dict[str, Any] = {}
    if patient_context_json:
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                ctx = parsed
        except json.JSONDecodeError:
            ctx = {}

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"invalid image: {e}") from e

    try:
        stage1 = run_medgemma_stage1(img, scores, ctx)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("MedGemma stage1 failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    _clamp_questions(stage1)

    sid_path = SESSION_DIR / f"{os.urandom(8).hex()}.bin"
    sid_path.write_bytes(raw)

    sess = _sessions.create(
        image_path=str(sid_path),
        pathology_scores=scores,
        patient_context=ctx,
        medgemma_stage1=stage1,
    )

    return {
        "session_id": sess.session_id,
        "follow_up_questions": stage1.get("follow_up_questions", []),
        "impression_draft": stage1.get("impression_draft", ""),
        "key_observations": stage1.get("key_observations", []),
        "uncertainties": stage1.get("uncertainties", []),
        "safety_flags": stage1.get("safety_flags", []),
        "models_used": ["google/medgemma-4b-it"],
    }


@app.post("/medgemma-cxr/session/complete")
async def session_complete(body: CompleteBody):
    sess = _sessions.get(body.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found or expired")

    qs = sess.medgemma_stage1.get("follow_up_questions") or []
    if not isinstance(qs, list):
        qs = []
    answers = normalize_answers_from_payload(qs, body.answers, body.skip_all)
    sess.answers = answers

    system_kimi = (
        "You are a senior radiology reporting assistant for Manthana Labs (India-aware). "
        "Produce one coherent report in Markdown. Do not repeat the interactive Q&A verbatim; "
        "integrate answers into Findings and Impression. Include a brief Limitations section."
    )
    user_kimi = build_kimi_user_payload(
        sess.pathology_scores,
        sess.patient_context,
        sess.medgemma_stage1,
        answers,
    )

    try:
        from llm_router import llm_router  # type: ignore

        jpeg_b64 = _jpeg_b64_from_path(sess.image_path)
        out = llm_router.complete_for_role(
            KIMI_REPORT_ROLE,
            system_kimi,
            user_kimi,
            image_b64=jpeg_b64,
            image_mime="image/jpeg",
            max_tokens=4096,
        )
        narrative = (out.get("content") or "").strip()
        model_used = out.get("model_used") or KIMI_REPORT_ROLE
    except Exception as e:
        logger.exception("Kimi final report failed")
        raise HTTPException(status_code=503, detail=f"final report generation failed: {e}") from e

    _sessions.delete(body.session_id)
    try:
        Path(sess.image_path).unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "session_id": body.session_id,
        "narrative_report": narrative,
        "answers_recorded": answers,
        "models_used": ["google/medgemma-4b-it", str(model_used)],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
