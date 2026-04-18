"""
Manthana — OpenRouter-only LLM router (SSOT: ../../config/cloud_inference.yaml).
Includes schema enforcement via instructor library and contradiction detection.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger("manthana.llm_router")


def _safe_unpack_chat_result(result) -> tuple[str, str, dict]:
    """Unpack chat_complete_sync result that may be a (text, model) 2-tuple
    (old manthana-inference) or a (text, model, usage) 3-tuple (current).
    Tolerates both so a stale Modal image layer never crashes the service.
    """
    logger.debug("chat_complete_sync returned: type=%s, len=%s, value=%s", type(result), len(result) if hasattr(result, '__len__') else 'N/A', result)
    if isinstance(result, (tuple, list)):
        if len(result) == 3:
            logger.debug("Unpacking 3-tuple as expected")
            return str(result[0]), str(result[1]), dict(result[2]) if result[2] else {}
        elif len(result) == 2:
            logger.error(
                "CRITICAL: chat_complete_sync returned 2-tuple (stale manthana-inference). "
                "This indicates Modal is using cached old version. Result: %s", result
            )
            return str(result[0]), str(result[1]), {}
        else:
            raise ValueError(f"Unexpected chat_complete_sync return length: {len(result)}")
    raise TypeError(f"Expected tuple/list from chat_complete_sync, got {type(result)}")


def _compute_repo_root() -> Path:
    """Resolve repo root containing packages/manthana-inference (this_studio) or set MANTHANA_LLM_REPO_ROOT."""
    env = (os.environ.get("MANTHANA_LLM_REPO_ROOT") or "").strip()
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        mi = parent / "packages" / "manthana-inference" / "src"
        if mi.is_dir():
            return parent
    # Railway report_assembly image: /app/shared/llm_router.py + /app/packages/manthana-inference
    p = Path(__file__).resolve()
    if p.parent.name == "shared":
        cand = p.parents[1] / "packages" / "manthana-inference" / "src"
        if cand.is_dir():
            return p.parents[1]
    # Legacy monorepo: manthana-backend/shared/llm_router.py -> .../this_studio
    try:
        return p.parents[3]
    except IndexError:
        try:
            return p.parents[2]
        except IndexError:
            return p.parent


_REPO_ROOT = _compute_repo_root()
_MI_SRC = _REPO_ROOT / "packages" / "manthana-inference" / "src"
if _MI_SRC.is_dir() and str(_MI_SRC) not in sys.path:
    sys.path.insert(0, str(_MI_SRC))

from manthana_inference import (  # type: ignore  # noqa: E402
    build_nim_sync_client,
    build_openrouter_sync_client,
    chat_complete_sync,
    load_cloud_inference_config,
    resolve_role,
)

# Version check to ensure we have the correct manthana-inference
try:
    import inspect
    sig = inspect.signature(chat_complete_sync)
    logger.info("manthana-inference chat_complete_sync signature: %s", sig)
    # Check if the function returns the correct tuple format by examining source
    source = inspect.getsource(chat_complete_sync)
    if "return text, model_eff, usage_info" in source:
        logger.info("manthana-inference has correct 3-tuple return format")
    else:
        logger.error("manthana-inference has outdated return format - this will cause tuple errors")
except Exception as e:
    logger.warning("Could not verify manthana-inference version: %s", e)

# Create a safe wrapper for chat_complete_sync that handles any tuple format
def safe_chat_complete_sync(
    client,
    config,
    role,
    messages,
    *,
    role_cfg=None,
    response_format=None,
    web_search_parameters=None,
    openrouter_api_keys=None,
):
    """Wrapper that ensures 3-tuple return regardless of manthana-inference version."""
    try:
        result = chat_complete_sync(
            client,
            config,
            role,
            messages,
            role_cfg=role_cfg,
            response_format=response_format,
            web_search_parameters=web_search_parameters,
            openrouter_api_keys=openrouter_api_keys,
        )
        logger.debug("chat_complete_sync returned: %s", result)
        if isinstance(result, tuple) and len(result) == 3:
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            logger.warning("chat_complete_sync returned 2-tuple, adding empty usage_info")
            return result[0], result[1], {}
        else:
            logger.error("chat_complete_sync returned unexpected format: %s", result)
            # Try to extract what we can
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1], {}
            else:
                raise ValueError(f"Cannot extract valid result from: {result}")
    except Exception as e:
        logger.error("chat_complete_sync failed completely: %s", e)
        raise

from schema_enforcement import (
    MODALITY_SCHEMAS,
    get_schema_for_modality,
    validate_llm_output,
)
from contradiction_detector import check_narrative_consistency

try:
    from instructor import from_openai
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.debug("instructor library not available; schema enforcement will be post-hoc")


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


def _nim_api_key() -> str:
    return (os.environ.get("NVIDIA_NIM_API_KEY") or "").strip()


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
        """
        Complete text with LLM via OpenRouter.
        force_model param is deprecated - models come from cloud_inference.yaml roles.
        """
        if force_model:
            logger.warning("force_model parameter is deprecated; models are configured via cloud_inference.yaml")

        cfg = _load_cfg()
        role = _TASK_TO_ROLE.get(task_type)
        if role is None:
            role = task_type if task_type in cfg.roles else "narrative_default"
        rc = resolve_role(cfg, role)
        rc = rc.model_copy(update={"temperature": temperature, "max_tokens": max_tokens})
        messages = self._build_messages(system_prompt, prompt)
        fmt = {"type": "json_object"} if requires_json else None

        if getattr(rc, "provider", "openrouter") == "nim":
            nim_key = _nim_api_key()
            if not nim_key:
                raise ValueError(
                    "NVIDIA_NIM_API_KEY is required for NIM provider roles (see config/cloud_inference.yaml)."
                )
            try:
                client = build_nim_sync_client(nim_key, cfg)
                raw = safe_chat_complete_sync(
                    client,
                    cfg,
                    role,
                    list(messages),
                    role_cfg=rc,
                    response_format=fmt,
                )
                try:
                    text, model_used, usage_info = _safe_unpack_chat_result(raw)
                except (ValueError, TypeError) as unpack_err:
                    logger.error("Failed to unpack chat result: %s. Raw result: %s", unpack_err, raw)
                    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
                        text, model_used = str(raw[0]), str(raw[1])
                        usage_info = {}
                    else:
                        raise unpack_err
                return {
                    "content": text,
                    "model_used": model_used,
                    "usage": usage_info or {},
                    "finish_reason": "stop",
                }
            except Exception as e:
                raise RuntimeError(f"NIM completion failed for role {role!r}: {e}") from e

        keys = _openrouter_keys()
        if not keys:
            raise ValueError(
                "No LLM configured. Set OPENROUTER_API_KEY (see config/cloud_inference.yaml)."
            )

        client0 = build_openrouter_sync_client(keys[0], cfg)
        raw = safe_chat_complete_sync(
            client0,
            cfg,
            role,
            list(messages),
            role_cfg=rc,
            response_format=fmt,
            openrouter_api_keys=keys,
        )
        try:
            text, model_used, usage_info = _safe_unpack_chat_result(raw)
        except (ValueError, TypeError) as unpack_err:
            logger.error("Failed to unpack chat result: %s. Raw result: %s", unpack_err, raw)
            if isinstance(raw, (tuple, list)) and len(raw) >= 2:
                text, model_used = str(raw[0]), str(raw[1])
                usage_info = {}
            else:
                raise unpack_err
        return {
            "content": text,
            "model_used": model_used,
            "usage": usage_info or {},
            "finish_reason": "stop",
        }

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
        image_b64_list: Optional[List[str]] = None,
        image_mime: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """
        Text or vision completion for an arbitrary SSOT role (e.g. narrative_mri, vision_primary).
        Supports single image (image_b64) or multiple images (image_b64_list) for film-photo mode.
        Returns dict with content, model_used, and usage information.
        """
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
        
        # Build multi-image content for film-photo mode
        if image_b64_list and len(image_b64_list) > 0:
            content_parts: List[Dict[str, Any]] = []
            for b64 in image_b64_list:
                if b64 and b64.strip():
                    url = f"data:{image_mime};base64,{b64.strip()}"
                    content_parts.append({"type": "image_url", "image_url": {"url": url}})
            content_parts.append({"type": "text", "text": user_c})
            user_content = content_parts
        elif image_b64 and image_b64.strip():
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

        if getattr(rc, "provider", "openrouter") == "nim":
            nim_key = _nim_api_key()
            if not nim_key:
                raise ValueError(
                    "NVIDIA_NIM_API_KEY is required for NIM provider roles (see config/cloud_inference.yaml)."
                )
            try:
                client = build_nim_sync_client(nim_key, cfg)
                raw = safe_chat_complete_sync(
                    client,
                    cfg,
                    role,
                    list(messages),
                    role_cfg=rc,
                    response_format=fmt,
                )
                text, model_used, usage_info = _safe_unpack_chat_result(raw)
                return {
                    "content": text,
                    "model_used": model_used,
                    "usage": usage_info or {},
                    "finish_reason": "stop",
                }
            except Exception as e:
                raise RuntimeError(f"NIM completion failed for role {role!r}: {e}") from e

        keys = _openrouter_keys()
        if not keys:
            raise ValueError(
                "No LLM configured. Set OPENROUTER_API_KEY (see config/cloud_inference.yaml)."
            )

        client0 = build_openrouter_sync_client(keys[0], cfg)
        logger.debug("OpenRouter call role=%s (multi-key model failover)", role)
        raw = safe_chat_complete_sync(
            client0,
            cfg,
            role,
            list(messages),
            role_cfg=rc,
            response_format=fmt,
            openrouter_api_keys=keys,
        )
        try:
            text, model_used, usage_info = _safe_unpack_chat_result(raw)
        except (ValueError, TypeError) as unpack_err:
            logger.error("Failed to unpack chat result in role %s: %s. Raw result: %s", role, unpack_err, raw)
            if isinstance(raw, (tuple, list)) and len(raw) >= 2:
                text, model_used = str(raw[0]), str(raw[1])
                usage_info = {}
            else:
                raise unpack_err
        return {
            "content": text,
            "model_used": model_used,
            "usage": usage_info or {},
            "finish_reason": "stop",
        }

    def complete_for_role_with_web_search(
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
        web_search_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Vision/text completion with optional OpenRouter openrouter:web_search server tool.
        web_search_parameters example:
          {"max_results": 3, "max_total_results": 6, "allowed_domains": ["pubmed.ncbi.nlm.nih.gov"]}
        """
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
        if image_b64 and image_b64.strip():
            url = f"data:{image_mime};base64,{image_b64.strip()}"
            user_content: Union[str, List[Dict[str, Any]]] = [
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

        if getattr(rc, "provider", "openrouter") == "nim":
            if web_search_parameters:
                logger.warning(
                    "NIM provider ignores web_search_parameters for role=%s (OpenRouter-only tool)",
                    role,
                )
            nim_key = _nim_api_key()
            if not nim_key:
                raise ValueError(
                    "NVIDIA_NIM_API_KEY is required for NIM provider roles (see config/cloud_inference.yaml)."
                )
            try:
                client = build_nim_sync_client(nim_key, cfg)
                raw = safe_chat_complete_sync(
                    client,
                    cfg,
                    role,
                    list(messages),
                    role_cfg=rc,
                    response_format=fmt,
                    web_search_parameters=None,
                )
                text, model_used, usage_info = _safe_unpack_chat_result(raw)
                return {
                    "content": text,
                    "model_used": model_used,
                    "usage": usage_info or {},
                    "finish_reason": "stop",
                }
            except Exception as e:
                raise RuntimeError(f"NIM completion failed for role {role!r}: {e}") from e

        keys = _openrouter_keys()
        if not keys:
            raise ValueError(
                "No LLM configured. Set OPENROUTER_API_KEY (see config/cloud_inference.yaml)."
            )

        client0 = build_openrouter_sync_client(keys[0], cfg)
        raw = safe_chat_complete_sync(
            client0,
            cfg,
            role,
            list(messages),
            role_cfg=rc,
            response_format=fmt,
            web_search_parameters=web_search_parameters,
            openrouter_api_keys=keys,
        )
        try:
            text, model_used, usage_info = _safe_unpack_chat_result(raw)
        except (ValueError, TypeError) as unpack_err:
            logger.error("Failed to unpack chat result in role %s: %s", role, unpack_err)
            if isinstance(raw, (tuple, list)) and len(raw) >= 2:
                text, model_used = str(raw[0]), str(raw[1])
                usage_info = {}
            else:
                raise unpack_err
        return {
            "content": text,
            "model_used": model_used,
            "usage": usage_info or {},
            "finish_reason": "stop",
        }

    def complete_with_schema(
        self,
        role: str,
        system_prompt: str,
        user_text: str,
        modality: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        image_b64: Optional[str] = None,
        image_b64_list: Optional[List[str]] = None,
        image_mime: str = "image/jpeg",
        automated_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete with schema enforcement and contradiction detection.
        Supports single image or multiple images (film-photo mode).
        
        Args:
            role: OpenRouter role from config
            system_prompt: System context
            user_text: User query/prompt
            modality: Imaging modality for schema selection
            automated_scores: Pipeline scores for contradiction detection
            
        Returns:
            Dict with structured output, validation results, and contradiction report
        """
        # Get base completion
        result = self.complete_for_role(
            role=role,
            system_prompt=system_prompt,
            user_text=user_text,
            temperature=temperature,
            max_tokens=max_tokens,
            requires_json=True,
            image_b64=image_b64,
            image_b64_list=image_b64_list,
            image_mime=image_mime,
        )
        
        content = result.get("content", "")
        structured_output = None
        validation_error = None
        
        # Try to parse and validate JSON
        try:
            parsed = json.loads(content)
            
            # Validate against modality schema
            is_valid, validated_or_error = validate_llm_output(parsed, modality)
            
            if is_valid:
                structured_output = validated_or_error.model_dump()
                result["schema_valid"] = True
            else:
                structured_output = parsed  # Return original even if invalid
                result["schema_valid"] = False
                result["schema_error"] = validated_or_error
        except json.JSONDecodeError as e:
            result["schema_valid"] = False
            result["schema_error"] = f"JSON parse error: {e}"
            structured_output = {"raw_text": content}
        
        # Run contradiction detection if automated scores provided
        if automated_scores:
            consistency = check_narrative_consistency(modality, automated_scores, content)
            result["contradiction_check"] = consistency
            result["contradictions_detected"] = not consistency["consistent"]
        
        result["structured_output"] = structured_output
        result["modality"] = modality
        return result

    def get_schema_for_modality(self, modality: str) -> Optional[Type]:
        """Get Pydantic schema class for a modality."""
        schema_class = get_schema_for_modality(modality)
        if schema_class.__name__ == "BaseModel":  # Default fallback
            return None
        return schema_class

    def get_model_info(self) -> dict:
        return {
            "strategy": self.strategy,
            "provider": "openrouter+nim",
            "nim_configured": bool(_nim_api_key()),
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
