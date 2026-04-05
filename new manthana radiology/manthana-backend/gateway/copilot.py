from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import verify_token

_GATEWAY_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_GATEWAY_DIR)
_shared = os.path.join(_BACKEND_ROOT, "shared")
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from llm_router import llm_router  # noqa: E402


router = APIRouter()


class CopilotRequest(BaseModel):
    question: str
    context: dict


COPILOT_SYSTEM = """
You are Manthana, an AI radiology assistant helping a clinician interpret an AI analysis result.

The JSON context you receive contains:
- findings (labels, severity, confidence, regions, descriptions)
- pathology_scores (per-pathology probabilities)
- narrative/impression text
- modality and detected_region
- patient context if available (age, sex, tobacco use, location, clinical notes).

Rules:
- Answer ONLY the clinician's specific question using the provided context.
- If the answer is not clearly supported by the findings or scores, say so explicitly and suggest what data would be needed.
- Be concise (ideally ≤150 words) unless a short bullet list is clearly better.
- Do not give definitive treatment instructions; instead, frame suggestions as options for the treating clinician.
- Assume Indian clinical context (higher TB prevalence, common tobacco use, cardiovascular risk in younger patients).
- Always end with: "This is AI assistance only — clinical judgement and local protocols must guide decisions."
""".strip()


@router.post("/copilot")
async def copilot_answer(
    req: CopilotRequest,
    token_data: dict = Depends(verify_token),
) -> dict[str, Any]:
    """Contextual Q&A over a single AnalysisResponse (OpenRouter / copilot role)."""
    try:
        context_str = json.dumps(req.context, indent=2)[:8000]
    except TypeError:
        context_str = str(req.context)[:8000]

    user_msg = (
        "ANALYSIS CONTEXT (JSON):\n"
        f"{context_str}\n\n"
        "CLINICIAN QUESTION:\n"
        f"{req.question.strip()}\n\n"
        "Answer strictly following the rules in the system prompt."
    )

    def _sync() -> str:
        r = llm_router.complete(
            prompt=user_msg,
            system_prompt=COPILOT_SYSTEM,
            task_type="copilot",
            max_tokens=600,
            temperature=0.3,
        )
        return (r.get("content") or "").strip()

    try:
        answer = await asyncio.to_thread(_sync)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable: {exc!s}",
        ) from exc

    if not answer:
        raise HTTPException(
            status_code=503,
            detail="LLM providers unavailable for CoPilot. Set OPENROUTER_API_KEY.",
        )

    return {"answer": answer}
