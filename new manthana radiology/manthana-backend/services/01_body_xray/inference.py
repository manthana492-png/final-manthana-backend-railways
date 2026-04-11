"""
CXR inference entrypoints for HTTP, ZeroClaw, and tests.

TorchXRayVision ensemble lives in pipeline_chest.run_chest_pipeline.
Optional narrative: OpenRouter only (SSOT: config/cloud_inference.yaml). Never raises to caller.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger("manthana.xray.inference")

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_CXR_SYSTEM = _PROMPT_DIR / "cxr_system.md"


def _load_api_keys_env() -> None:
    """Load workspace api-keys.env into os.environ if present (keys not already set)."""
    here = Path(__file__).resolve()
    for p in list(here.parents)[:10]:
        candidate = p / "api-keys.env"
        if not candidate.is_file():
            continue
        try:
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key:
                        os.environ.setdefault(key, val)
            logger.debug("Loaded env keys from %s", candidate)
        except OSError as e:
            logger.warning("Could not read %s: %s", candidate, e)
        return


_load_api_keys_env()

XRAY_REQUIRE_LLM_NARRATIVE = os.getenv(
    "XRAY_REQUIRE_LLM_NARRATIVE",
    os.getenv("XRAY_REQUIRE_KIMI_NARRATIVE", "1"),
).strip().lower() in ("1", "true", "yes")
# Legacy name (env XRAY_REQUIRE_KIMI_NARRATIVE still supported above); tests may patch this alias.
XRAY_REQUIRE_KIMI_NARRATIVE = XRAY_REQUIRE_LLM_NARRATIVE


def run_pipeline(
    filepath: str,
    job_id: str,
    patient_context: dict | None = None,
    image_b64: str | None = None,
    *,
    skip_llm_narrative: bool = False,
) -> dict:
    """
    Run chest X-ray analysis. Optional LLM narrative uses OpenRouter from env.
    When ``skip_llm_narrative`` is True, returns TorchXRayVision + structured output only
    (for the MedGemma middle-layer on a separate Modal app).
    """
    path = filepath
    if image_b64:
        path = _write_b64_to_temp(image_b64, suffix=".png")

    image_b64_for_llm = image_b64
    if not image_b64_for_llm and filepath and os.path.isfile(filepath):
        try:
            with open(filepath, "rb") as f:
                image_b64_for_llm = base64.b64encode(f.read()).decode()
        except OSError:
            image_b64_for_llm = None

    try:
        from pipeline_chest import run_chest_pipeline

        out = run_chest_pipeline(path, job_id)
        out["job_id"] = job_id
    finally:
        if image_b64 and path != filepath and path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass

    if not skip_llm_narrative:
        out = attach_narrative(
            out,
            patient_context=patient_context,
            image_b64=image_b64_for_llm,
        )

    return out


def run_cxr_pipeline_b64(
    image_b64: str,
    patient_context: str = "",
    job_id: str = "",
) -> dict:
    """Base64 image → temp file → run_chest_pipeline (for ZeroClaw / agents)."""
    jid = job_id or str(uuid.uuid4())[:8]
    ctx: dict | None = None
    if patient_context:
        try:
            import json

            ctx = json.loads(patient_context) if isinstance(patient_context, str) else patient_context
        except Exception:
            ctx = None
    return run_pipeline(
        filepath="",
        job_id=jid,
        patient_context=ctx,
        image_b64=image_b64,
        skip_llm_narrative=False,
    )


run_chest_xray_pipeline_b64 = run_cxr_pipeline_b64


def _write_b64_to_temp(image_b64: str, suffix: str = ".png") -> str:
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid base64 image: {e}") from e
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="cxr_b64_")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _to_jpeg_b64(image_b64: str) -> str:
    from PIL import Image

    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _read_system_prompt() -> str:
    try:
        if _CXR_SYSTEM.is_file():
            return _CXR_SYSTEM.read_text(encoding="utf-8")[:12000]
    except OSError:
        pass
    return (
        "You are a senior chest radiologist. Summarise findings clearly and conservatively. "
        "Do not invent pathology scores; use only the JSON provided."
    )


def _optional_llm_narrative(
    pathology_scores: dict,
    impression: str,
    patient_context: dict | None,
    image_b64: str | None = None,
) -> str:
    """OpenRouter vision (if image) then text. Never raises."""
    import json

    scores_json = json.dumps(pathology_scores, indent=2)[:12000]
    patient_json = json.dumps(patient_context or {}, indent=2)[:4000]
    user_text = (
        f"IMPRESSION (model):\n{impression}\n\n"
        f"PATHOLOGY_SCORES (0–1, TorchXRayVision ensemble):\n{scores_json}\n\n"
        f"PATIENT_CONTEXT:\n{patient_json}\n\n"
        "Write a concise radiology-style paragraph (India: TB endemicity, occupational exposure when relevant). "
        "Do not contradict the numeric scores."
    )
    system = _read_system_prompt()

    try:
        from llm_router import llm_router
    except Exception as exc:
        logger.warning("CXR narrative: llm_router unavailable: %s", exc)
        return ""

    if image_b64:
        try:
            jpeg_b64 = _to_jpeg_b64(image_b64)
            out = llm_router.complete_for_role(
                "vision_primary",
                system,
                user_text,
                image_b64=jpeg_b64,
                image_mime="image/jpeg",
                max_tokens=1500,
            )
            txt = (out.get("content") or "").strip()
            if txt:
                return txt
        except Exception as exc:
            logger.warning("OpenRouter CXR vision narrative failed: %s", exc)

    try:
        out = llm_router.complete_for_role(
            "narrative_default",
            system,
            user_text,
            max_tokens=1200,
        )
        return (out.get("content") or "").strip()
    except Exception as exc:
        logger.warning("OpenRouter CXR text narrative failed: %s", exc)
        return ""


def attach_narrative(
    result: dict,
    patient_context: dict | None = None,
    image_b64: str | None = None,
) -> dict:
    """
    Attach OpenRouter narrative report to any X-ray result with dict structures.
    """
    st = result.get("structures")
    if not isinstance(st, dict):
        return result
    narrative = _optional_llm_narrative(
        pathology_scores=result.get("pathology_scores") or {},
        impression=result.get("impression") or "",
        patient_context=patient_context,
        image_b64=image_b64,
    )
    if narrative:
        st["narrative_report"] = narrative
        result["structures"] = st
        models = result.get("models_used")
        if isinstance(models, list) and "OpenRouter-narrative-CXR" not in models:
            models.append("OpenRouter-narrative-CXR")
            result["models_used"] = models
    elif XRAY_REQUIRE_KIMI_NARRATIVE:
        raise RuntimeError("LLM narrative is required for X-ray but generation failed.")
    return result
