"""
95-modality AI orchestration: /ai/detect-modality, /ai/interrogate, /ai/interpret
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from auth import verify_token

logger = logging.getLogger("manthana.ai_orchestrator")

router = APIRouter(prefix="/ai", tags=["ai-orchestration"])

# OpenRouter server-tool params (Exa / native); passed to manthana_inference chat_complete_sync
WEB_SEARCH_PARAMETERS: Dict[str, Any] = {
    "max_results": 3,
    "max_total_results": 6,
    "allowed_domains": [
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "nature.com",
        "nejm.org",
        "thelancet.com",
        "rsna.org",
        "arxiv.org",
        "radiopaedia.org",
    ],
}


def _orch_enabled() -> bool:
    return (os.getenv("AI_ORCHESTRATION_ENABLED", "true") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _require_orch() -> None:
    if not _orch_enabled():
        raise HTTPException(status_code=503, detail="AI orchestration is disabled")


def _strip_json_fence(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _parse_json_obj(s: str) -> Dict[str, Any]:
    raw = _strip_json_fence(s)
    return json.loads(raw)


# --- Lazy imports (gateway path) ---
def _llm():
    from llm_router import llm_router

    return llm_router


def _registry():
    from modality_registry import (
        AUTO_DETECT_KEY,
        MODALITY_GROUPS_META,
        MODALITY_REGISTRY,
        get_modality_config,
        list_modalities_for_prompt,
        validate_modality_key,
    )

    return (
        AUTO_DETECT_KEY,
        MODALITY_GROUPS_META,
        MODALITY_REGISTRY,
        get_modality_config,
        list_modalities_for_prompt,
        validate_modality_key,
    )


def _prompts():
    from prompts.interpreter_base import interpret_user_text_block, interpreter_system_prompt
    from prompts.interrogator_base import detect_modality_system_prompt, interrogator_system_prompt
    from prompts.modality_prompts import group_specialization_for

    return (
        interpret_user_text_block,
        interpreter_system_prompt,
        detect_modality_system_prompt,
        interrogator_system_prompt,
        group_specialization_for,
    )


def _session():
    from session_store import session_store

    return session_store


class DetectRequest(BaseModel):
    image_b64: Optional[str] = None
    image_mime: str = Field(default="image/jpeg")
    text_context: Optional[str] = Field(
        default=None,
        description="Optional raw text for report-only classification",
    )


class InterrogateRequest(BaseModel):
    image_b64: Optional[str] = None
    image_mime: str = "image/jpeg"
    modality_key: str
    patient_context_json: Optional[str] = None


class AnswerItem(BaseModel):
    question_id: str
    answer: str


class InterpretRequest(BaseModel):
    session_id: str
    answers: List[AnswerItem]
    patient_context_json: Optional[str] = None


@router.get("/modalities")
async def list_modalities(token_data: dict = Depends(verify_token)):
    """Registry for UI (95 modalities + groups)."""
    _require_orch()
    _, MODALITY_GROUPS_META, MODALITY_REGISTRY, _, _, _ = _registry()
    items = []
    for k, cfg in sorted(MODALITY_REGISTRY.items(), key=lambda x: (x[1].group, x[0])):
        items.append(
            {
                "key": k,
                "group": cfg.group,
                "display_name": cfg.display_name,
                "input_formats": list(cfg.input_formats),
                "interrogator_role": cfg.interrogator_role,
                "interpreter_role": cfg.interpreter_role,
            }
        )
    return {"groups": MODALITY_GROUPS_META, "modalities": items}


@router.post("/detect-modality")
async def detect_modality(
    body: DetectRequest,
    token_data: dict = Depends(verify_token),
    x_subscription_tier: Optional[str] = Header(None, alias="X-Subscription-Tier"),
):
    _require_orch()
    (
        _auto,
        _gm,
        _reg,
        _get,
        list_for_prompt,
        _val,
    ) = _registry()
    (
        _iub,
        _interp_sys,
        detect_sys,
        _iq_sys,
        _gs,
    ) = _prompts()

    if not body.image_b64 and not (body.text_context or "").strip():
        raise HTTPException(status_code=400, detail="image_b64 or text_context required")

    llm = _llm()
    modality_list = list_for_prompt()
    system = detect_modality_system_prompt(modality_list_text=modality_list)
    user_text = "Classify this study."
    if body.text_context and not body.image_b64:
        user_text = f"Document text (excerpt):\n{(body.text_context or '')[:12000]}"

    try:
        if body.image_b64:
            r = llm.complete_for_role(
                role="modality_detect",
                system_prompt=system,
                user_text=user_text,
                requires_json=True,
                image_b64=body.image_b64,
                image_mime=body.image_mime or "image/jpeg",
            )
        else:
            r = llm.complete_for_role(
                role="modality_detect",
                system_prompt=system,
                user_text=user_text,
                requires_json=True,
            )
        data = _parse_json_obj(r.get("content") or "{}")
    except Exception as e:
        logger.exception("detect-modality failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {
        "modality_key": data.get("modality_key", "unknown"),
        "confidence": float(data.get("confidence") or 0),
        "group": data.get("group"),
        "reason": data.get("reason"),
        "model_used": r.get("model_used"),
    }


@router.post("/interrogate")
async def interrogate(
    body: InterrogateRequest,
    token_data: dict = Depends(verify_token),
):
    _require_orch()
    _, _, _, get_modality_config, _, validate_modality_key = _registry()
    (
        _iub,
        _interp_sys,
        _det,
        interrogator_system_prompt,
        group_spec,
    ) = _prompts()
    store = _session()

    mk = (body.modality_key or "").strip().lower()
    if mk in ("auto", ""):
        raise HTTPException(status_code=400, detail="modality_key is required (use /detect-modality first for auto)")
    if not validate_modality_key(mk):
        raise HTTPException(status_code=422, detail=f"Unknown modality_key: {mk}")

    cfg = get_modality_config(mk)
    if not cfg:
        raise HTTPException(status_code=422, detail="Invalid modality")

    sys_p = interrogator_system_prompt(modality_key=cfg.key, display_name=cfg.display_name)
    user_t = "Generate questions JSON as specified."
    lr = _llm()
    try:
        if body.image_b64:
            r = lr.complete_for_role(
                role=cfg.interrogator_role,
                system_prompt=sys_p,
                user_text=user_t,
                requires_json=True,
                image_b64=body.image_b64,
                image_mime=body.image_mime or "image/jpeg",
            )
        else:
            # text-only modalities
            r = lr.complete_for_role(
                role=cfg.interrogator_role,
                system_prompt=sys_p,
                user_text=(body.patient_context_json or "No extra text.")[:16000],
                requires_json=True,
            )
        qdata = _parse_json_obj(r.get("content") or "{}")
        questions = qdata.get("questions") or []
    except Exception as e:
        logger.exception("interrogate failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    sid = store.create(
        image_b64=body.image_b64,
        image_mime=body.image_mime,
        modality_key=cfg.key,
        display_name=cfg.display_name,
        group=cfg.group,
        questions=questions,
        interrogator_model=str(r.get("model_used")),
        interpreter_role=cfg.interpreter_role,
        patient_context_json=body.patient_context_json,
    )

    return {
        "session_id": sid,
        "questions": questions,
        "model_used": r.get("model_used"),
        "modality_key": cfg.key,
    }


@router.post("/interpret")
async def interpret(
    body: InterpretRequest,
    token_data: dict = Depends(verify_token),
    x_subscription_tier: Optional[str] = Header(None, alias="X-Subscription-Tier"),
):
    _require_orch()
    _, _, _, get_modality_config, _, _ = _registry()
    (
        interpret_user_text_block,
        interpreter_system_prompt,
        _d,
        _iq,
        group_specialization_for,
    ) = _prompts()
    store = _session()

    row = store.get(body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session expired or not found")

    mk = row["modality_key"]
    cfg = get_modality_config(mk)
    if not cfg:
        raise HTTPException(status_code=500, detail="Session modality invalid")

    tier = (x_subscription_tier or "free").strip().lower()
    enable_web = tier not in ("", "free", "trial")

    qa_json = json.dumps([a.model_dump() for a in body.answers], ensure_ascii=False)
    pc = body.patient_context_json or row.get("patient_context_json")

    sys_p = interpreter_system_prompt(
        modality_key=cfg.key,
        display_name=cfg.display_name,
        group=cfg.group,
        group_extra=group_specialization_for(cfg.group),
        enable_web_search=enable_web,
    )
    user_t = interpret_user_text_block(qa_pairs_json=qa_json, patient_context_json=pc)

    llm = _llm()
    try:
        img = row.get("image_b64")
        mime = row.get("image_mime") or "image/jpeg"

        if enable_web:
            r = llm.complete_for_role_with_web_search(
                role=cfg.interpreter_role,
                system_prompt=sys_p,
                user_text=user_t,
                requires_json=True,
                image_b64=img,
                image_mime=mime,
                web_search_parameters=WEB_SEARCH_PARAMETERS,
            )
        else:
            # Free tier: no web search; use interpreter_free role from YAML
            r = llm.complete_for_role(
                role="interpreter_free",
                system_prompt=sys_p,
                user_text=user_t,
                requires_json=True,
                image_b64=img,
                image_mime=mime,
            )

        report = _parse_json_obj(r.get("content") or "{}")
        store.delete(body.session_id)
    except Exception as e:
        logger.exception("interpret failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    report.setdefault("models_used", [])
    if isinstance(report["models_used"], list):
        report["models_used"] = list(report["models_used"]) + [str(r.get("model_used"))]

    return {
        "report": report,
        "model_used": r.get("model_used"),
        "web_search_enabled": enable_web,
        "usage": r.get("usage") or {},
    }
