from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import verify_token


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


async def _try_kimi(system: str, user_msg: str) -> Optional[str]:
    """
    Call Moonshot / Kimi via OpenAI-compatible /chat/completions.
    Returns text content or None on failure.
    """
    api_key = (
        os.environ.get("KIMI_API_KEY", "").strip()
        or os.environ.get("MOONSHOT_API_KEY", "").strip()
    )
    if not api_key:
        return None

    base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
    model = os.environ.get("KIMI_MODEL", "kimi-k2.5").strip()

    extra_body: dict[str, Any] | None = None
    if "kimi-k2" in model.lower():
        extra_body = {"thinking": {"type": "disabled"}}

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    if extra_body is not None:
        payload["extra_body"] = extra_body

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        # OpenAI-style content might also be list[dict]
        if isinstance(content, list):
            parts = []
            for part in content:
                txt = part.get("text") if isinstance(part, dict) else None
                if isinstance(txt, str):
                    parts.append(txt)
            return "\n".join(parts).strip() or None
        return None
    except Exception:
        return None


async def _try_claude(system: str, user_msg: str) -> Optional[str]:
    """
    Call Anthropic Claude messages API directly.
    Returns text content or None on failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022").strip()
    url = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 600,
        "system": system,
        "messages": [
            {
                "role": "user",
                "content": user_msg,
            }
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        blocks = data.get("content") or []
        texts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text")
                if isinstance(txt, str):
                    texts.append(txt)
        out = "\n".join(texts).strip()
        return out or None
    except Exception:
        return None


@router.post("/copilot")
async def copilot_answer(
    req: CopilotRequest,
    token_data: dict = Depends(verify_token),
) -> dict:
    """
    Contextual Q&A over a single AnalysisResponse.
    Authenticated via JWT at the gateway.
    """
    try:
        context_str = json.dumps(req.context, indent=2)[:8000]
    except TypeError:
        # Fallback if context is not fully JSON-serializable
        context_str = str(req.context)[:8000]

    user_msg = (
        "ANALYSIS CONTEXT (JSON):\n"
        f"{context_str}\n\n"
        "CLINICIAN QUESTION:\n"
        f"{req.question.strip()}\n\n"
        "Answer strictly following the rules in the system prompt."
    )

    answer = await _try_kimi(COPILOT_SYSTEM, user_msg)
    if not answer:
        answer = await _try_claude(COPILOT_SYSTEM, user_msg)

    if not answer:
        raise HTTPException(
            status_code=503,
            detail="LLM providers unavailable for CoPilot. Try again later.",
        )

    return {"answer": answer}

