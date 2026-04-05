"""
Manthana — OpenRouter-only LLM router (SSOT: ../../config/cloud_inference.yaml).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("manthana.llm_router")

# Repo root: manthana-backend/shared/llm_router.py -> parents[3] = this_studio
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MI_SRC = _REPO_ROOT / "packages" / "manthana-inference" / "src"
if str(_MI_SRC) not in sys.path:
    sys.path.insert(0, str(_MI_SRC))

from manthana_inference import (  # type: ignore  # noqa: E402
    build_openrouter_sync_client,
    chat_complete_sync,
    load_cloud_inference_config,
    resolve_role,
)


class TaskType:
    LAB_REPORT = "lab_report"
    UNIFIED_REPORT = "unified_report"
    CLINICAL_QA = "clinical_qa"
    SUMMARIZATION = "summarization"
    CORRELATION = "correlation"
    COPILOT = "copilot"
    FALLBACK = "fallback"


_TASK_TO_ROLE = {
    TaskType.LAB_REPORT: "lab_report",
    TaskType.UNIFIED_REPORT: "unified_report",
    TaskType.CLINICAL_QA: "clinical_qa",
    TaskType.SUMMARIZATION: "summarization",
    TaskType.CORRELATION: "correlation",
    TaskType.COPILOT: "copilot",
    TaskType.FALLBACK: "fallback",
}


def _openrouter_keys() -> List[str]:
    keys: List[str] = []
    for env_name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"):
        k = (os.environ.get(env_name) or "").strip()
        if k and len(k) >= 8 and k not in keys:
            keys.append(k)
    return keys


def _load_cfg():
    path = (os.environ.get("CLOUD_INFERENCE_CONFIG_PATH") or "").strip()
    return load_cloud_inference_config(Path(path) if path else (_REPO_ROOT / "config" / "cloud_inference.yaml"))


class LLMRouter:
    """Routes all text completions through OpenRouter using YAML roles."""

    def __init__(self) -> None:
        self.strategy = os.getenv("LLM_ROUTING_STRATEGY", "openrouter")

    @staticmethod
    def _build_messages(system_prompt: str, prompt: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "clinical_qa",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        requires_json: bool = False,
        force_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = force_model  # reserved — models come from cloud_inference.yaml
        keys = _openrouter_keys()
        if not keys:
            raise ValueError(
                "No LLM configured. Set OPENROUTER_API_KEY (see config/cloud_inference.yaml)."
            )
        cfg = _load_cfg()
        role = _TASK_TO_ROLE.get(task_type)
        if role is None:
            role = task_type if task_type in cfg.roles else "narrative_default"
        rc = resolve_role(cfg, role)
        rc = rc.model_copy(update={"temperature": temperature, "max_tokens": max_tokens})
        messages = self._build_messages(system_prompt, prompt)
        fmt = {"type": "json_object"} if requires_json else None
        last_err: Optional[Exception] = None
        for api_key in keys:
            try:
                client = build_openrouter_sync_client(api_key, cfg)
                text, model_used = chat_complete_sync(
                    client,
                    cfg,
                    role,
                    list(messages),
                    role_cfg=rc,
                    response_format=fmt,
                )
                return {
                    "content": text,
                    "model_used": model_used,
                    "usage": {},
                    "finish_reason": "stop",
                }
            except Exception as e:
                last_err = e
                logger.warning("OpenRouter attempt failed: %s", e)
        raise RuntimeError(f"All OpenRouter keys failed. Last error: {last_err}")

    def complete_for_role(
        self,
        role: str,
        system_prompt: str,
        user_text: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        requires_json: bool = False,
        image_b64: Optional[str] = None,
        image_mime: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """
        Text or vision completion for an arbitrary SSOT role (e.g. narrative_mri, vision_primary).
        """
        keys = _openrouter_keys()
        if not keys:
            raise ValueError(
                "No LLM configured. Set OPENROUTER_API_KEY (see config/cloud_inference.yaml)."
            )
        cfg = _load_cfg()
        rc = resolve_role(cfg, role)
        upd: Dict[str, Any] = {}
        if temperature is not None:
            upd["temperature"] = temperature
        if max_tokens is not None:
            upd["max_tokens"] = max_tokens
        if upd:
            rc = rc.model_copy(update=upd)

        sys_c = (system_prompt or "")[:200000]
        user_c = user_text[:200000]
        user_content: Union[str, List[Dict[str, Any]]]
        if image_b64 and image_b64.strip():
            url = f"data:{image_mime};base64,{image_b64.strip()}"
            user_content = [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": user_c},
            ]
        else:
            user_content = user_c

        messages: List[Dict[str, Any]] = []
        if sys_c.strip():
            messages.append({"role": "system", "content": sys_c})
        messages.append({"role": "user", "content": user_content})

        fmt: Optional[Dict[str, Any]] = {"type": "json_object"} if requires_json else None
        last_err: Optional[Exception] = None
        for api_key in keys:
            try:
                client = build_openrouter_sync_client(api_key, cfg)
                text, model_used = chat_complete_sync(
                    client,
                    cfg,
                    role,
                    list(messages),
                    role_cfg=rc,
                    response_format=fmt,
                )
                return {
                    "content": text,
                    "model_used": model_used,
                    "usage": {},
                    "finish_reason": "stop",
                }
            except Exception as e:
                last_err = e
                logger.warning("OpenRouter role=%s attempt failed: %s", role, e)
        raise RuntimeError(f"All OpenRouter keys failed for role {role!r}. Last error: {last_err}")

    def get_model_info(self) -> dict:
        return {
            "strategy": self.strategy,
            "provider": "openrouter",
            "config": str(_REPO_ROOT / "config" / "cloud_inference.yaml"),
        }


llm_router = LLMRouter()


def complete_for_role(
    role: str,
    system_prompt: str,
    user_text: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Module-level helper for modality services."""
    return llm_router.complete_for_role(role, system_prompt, user_text, **kwargs)


def analyze_lab_report(structured_data: dict, raw_text: str) -> dict:
    prompt = f"""Analyze this structured lab report data:

STRUCTURED DATA:
{json.dumps(structured_data, indent=2, default=str)[:6000]}

RAW TEXT:
{raw_text[:3000]}

Provide clinical interpretation with severity classification."""
    return llm_router.complete(
        prompt=prompt,
        system_prompt="You are a senior clinical pathologist. Identify abnormalities and classify severity.",
        task_type=TaskType.LAB_REPORT,
        requires_json=True,
    )


def generate_unified_report(individual_results: list, patient_id: str) -> dict:
    results_json = json.dumps(individual_results, indent=2, default=str)
    prompt = f"""Synthesize these individual modality analyses into a unified clinical report:

PATIENT ID: {patient_id}

INDIVIDUAL RESULTS:
{results_json[:15000]}

Generate a comprehensive unified report that:
1. Identifies cross-modal correlations
2. Highlights critical findings
3. Provides integrated impression
4. Suggests follow-up recommendations"""
    return llm_router.complete(
        prompt=prompt,
        system_prompt="You are a senior radiologist synthesizing multi-modal findings.",
        task_type=TaskType.UNIFIED_REPORT,
        temperature=0.2,
        max_tokens=8192,
    )


def clinical_qa(question: str, context: dict) -> dict:
    context_json = json.dumps(context, indent=2, default=str)
    prompt = f"""Context:
{context_json[:4000]}

Question: {question}

Provide a clear, evidence-based answer."""
    return llm_router.complete(
        prompt=prompt,
        system_prompt="You are a knowledgeable medical AI assistant.",
        task_type=TaskType.CLINICAL_QA,
    )
