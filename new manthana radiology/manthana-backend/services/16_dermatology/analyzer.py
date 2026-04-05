"""
Manthana Dermatology Engine — OpenRouter vision (DermAI system prompt; SSOT config/cloud_inference.yaml)
+ structured scores + narrative.
V2: optional EfficientNet-B4 checkpoint replaces the score-extraction API call only.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from json import JSONDecodeError

from PIL import Image

# manthana-backend/shared
_root = Path(__file__).resolve().parent.parent.parent
if str(_root / "shared") not in sys.path:
    sys.path.insert(0, str(_root / "shared"))

from disclaimer import DISCLAIMER

from classifier import DERM_CLASSES
from critical_flags import check_derm_critical

logger = logging.getLogger("manthana.dermatology")

DERMAI_SYSTEM_PATH = Path(__file__).parent / "prompts" / "dermatology_dermai_system.txt"

_derm_classifier: Any = None
_dermai_system_text: str | None = None


def _try_load_classifier() -> None:
    """Load V2 classifier only if weights exist; never blocks or raises."""
    global _derm_classifier
    if _derm_classifier is not None:
        return
    try:
        from config import CHECKPOINT_FILENAME, DEVICE, MODEL_DIR

        from classifier import DermClassifier

        mp = Path(MODEL_DIR) / CHECKPOINT_FILENAME
        if mp.is_file():
            _derm_classifier = DermClassifier(mp, DEVICE)
            logger.info("DermClassifier loaded from %s", mp)
    except Exception as e:
        logger.debug("V2 classifier not loaded: %s", e)


def _load_dermai_system_prompt() -> str:
    global _dermai_system_text
    if _dermai_system_text is None:
        _dermai_system_text = DERMAI_SYSTEM_PATH.read_text(encoding="utf-8")
    return _dermai_system_text


def _sniff_media_type(image_bytes: bytes) -> str:
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"


def _vision_b64_and_mime(image_bytes: bytes) -> tuple[str, str]:
    """Raw base64 + mime for OpenRouter vision."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
    except Exception:
        b64 = base64.b64encode(image_bytes).decode()
        return b64, _sniff_media_type(image_bytes)


def _openrouter_derm_complete(
    *,
    system_prompt: str,
    user_text: str,
    image_b64: str | None,
    image_mime: str,
    max_tokens: int,
    requires_json: bool,
    call_label: str,
) -> tuple[str, str]:
    """Returns (content, model_used_slug). Raises on failure."""
    from llm_router import llm_router

    t0 = time.perf_counter()
    out = llm_router.complete_for_role(
        "dermatology",
        system_prompt,
        user_text,
        max_tokens=max_tokens,
        requires_json=requires_json,
        image_b64=image_b64,
        image_mime=image_mime,
    )
    elapsed = time.perf_counter() - t0
    logger.debug("%s: OpenRouter complete_for_role done in %.2fs", call_label, elapsed)
    text = (out.get("content") or "").strip()
    mu = str(out.get("model_used") or "").strip()
    return text, mu


def _parse_json_from_llm(text: str) -> dict[str, Any]:
    """Strip fences; parse JSON; if model adds prose, extract first JSON object."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    t = t.strip()
    try:
        out = json.loads(t)
        if isinstance(out, dict):
            return out
    except JSONDecodeError:
        pass
    start = t.find("{")
    if start == -1:
        raise JSONDecodeError("No JSON object found in model output", text, 0)
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(t[start:])
    if not isinstance(obj, dict):
        raise JSONDecodeError("Parsed JSON is not an object", text, start)
    return obj


def _normalize_condition_scores(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in DERM_CLASSES:
        v = raw.get(k)
        try:
            out[k] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            out[k] = 0.0
    s = sum(out[k] for k in DERM_CLASSES)
    if s > 0 and abs(s - 1.0) > 0.2:
        for k in DERM_CLASSES:
            out[k] = round(out[k] / s, 4)
    tc = raw.get("top_class") or max(out, key=out.get)
    if tc not in DERM_CLASSES:
        tc = max(out, key=out.get)
    conf = float(raw.get("confidence", out.get(tc, 0.0)) or 0.0)
    cl = raw.get("confidence_label", "low")
    if cl not in ("high", "medium", "low"):
        cl = "high" if conf >= 0.70 else "medium" if conf >= 0.45 else "low"
    out["top_class"] = tc
    out["confidence"] = conf
    out["confidence_label"] = cl
    out["is_malignant_candidate"] = bool(
        raw.get("is_malignant_candidate", tc in {"bcc", "scc", "melanoma"})
    )
    return out


def _get_openrouter_condition_scores(
    patient_context: dict[str, Any],
    system_prompt: str,
    image_bytes: bytes,
) -> tuple[dict[str, Any], str]:
    """OpenRouter vision — JSON condition scores for pipeline compatibility."""
    score_user = (
        "Apply your full diagnostic framework (Steps 1–5) for this skin photograph.\n\n"
        "Your final assistant message must be ONLY valid JSON (no markdown fences, no commentary) "
        "with probability scores for each condition from 0.0 to 1.0 (approximately summing to 1.0).\n\n"
        "Required keys exactly:\n"
        "  tinea, vitiligo, psoriasis, melasma, acne, eczema_dermatitis, scabies, urticaria,\n"
        "  bcc, scc, melanoma, normal_benign,\n"
        "  top_class, confidence, confidence_label (one of high|medium|low), "
        "is_malignant_candidate (boolean).\n\n"
        "Patient context (may be empty): "
        + json.dumps(patient_context, ensure_ascii=False)
    )
    b64, mime = _vision_b64_and_mime(image_bytes)
    text, model_used = _openrouter_derm_complete(
        system_prompt=system_prompt,
        user_text=score_user,
        image_b64=b64,
        image_mime=mime,
        max_tokens=int(os.getenv("DERM_SCORE_MAX_TOKENS", os.getenv("KIMI_DERMATOLOGY_SCORE_MAX_TOKENS", "8192"))),
        requires_json=True,
        call_label="derm_scores",
    )
    raw = _parse_json_from_llm(text)
    return _normalize_condition_scores(raw), model_used


def _build_findings(
    condition_scores: dict[str, Any], critical: dict[str, Any]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if critical.get("is_critical"):
        findings.append(
            {
                "label": "Possible malignancy (screening alert)",
                "severity": "critical",
                "confidence": min(
                    100.0,
                    float(critical.get("top_malignancy_score", 0.0) or 0.0) * 100.0,
                ),
                "description": critical.get("action", "Urgent dermatology referral."),
            }
        )
    top3 = sorted(
        ((k, float(condition_scores[k])) for k in DERM_CLASSES if k in condition_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    for cond, score in top3:
        sev = "warning" if score >= 0.45 else "info"
        findings.append(
            {
                "label": cond.replace("_", " ").title(),
                "severity": sev,
                "confidence": round(score * 100.0, 1),
                "description": f"Model-estimated probability {score:.2%}.",
            }
        )
    return findings


def _extract_impression(report_text: str) -> str:
    m = re.search(
        r"IMPRESSION[:\s]+(.+?)(?:\n{2,}|\n(?:\d+\.|#|\*\*|══))",
        report_text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()[:500]
    return report_text[:400].strip()


def analyze_dermatology(
    image_b64: str,
    patient_context: dict[str, Any],
    job_id: str = "",
) -> dict[str, Any]:
    """
    Full pipeline. Returns a dict suitable for AnalysisResponse(**d).
    Two OpenRouter calls when scores come from vision; one OpenRouter call if V2 classifier supplies scores.
    """
    _try_load_classifier()

    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = pil_image.size
    from config import DERM_MAX_IMAGE_PIXELS

    if w * h > DERM_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image too large ({w}×{h} pixels); max {DERM_MAX_IMAGE_PIXELS} pixels"
        )

    system_prompt = _load_dermai_system_prompt()
    score_model_slug = ""

    if _derm_classifier is not None:
        condition_scores = _derm_classifier.classify(pil_image)
        classifier_mode = "efficientnet_b4"
    else:
        condition_scores, score_model_slug = _get_openrouter_condition_scores(
            patient_context, system_prompt, image_bytes
        )
        classifier_mode = "openrouter_vision_v1"

    critical = check_derm_critical(condition_scores)

    measurements_json = json.dumps(
        {
            "condition_scores": {k: condition_scores.get(k) for k in DERM_CLASSES}
            | {
                "top_class": condition_scores.get("top_class"),
                "confidence": condition_scores.get("confidence"),
                "confidence_label": condition_scores.get("confidence_label"),
                "is_malignant_candidate": condition_scores.get("is_malignant_candidate"),
            },
            "critical_flag": critical,
            "patient_context": patient_context,
            "classifier_mode": classifier_mode,
        },
        indent=2,
        ensure_ascii=False,
    )

    narr_b64, narr_mime = _vision_b64_and_mime(image_bytes)
    narrative_user_text = (
        f"QUANTITATIVE MEASUREMENTS:\n{measurements_json}\n\n"
        "Generate the dermatology assessment report following your instructions "
        "(morphology, differential, India context, safety). "
        "If your instructions include a JSON structured block for narrative output, you may use it; "
        "otherwise use clear section headings including IMPRESSION."
    )
    report_text, narr_model_slug = _openrouter_derm_complete(
        system_prompt=system_prompt,
        user_text=narrative_user_text,
        image_b64=narr_b64,
        image_mime=narr_mime,
        max_tokens=int(os.getenv("DERM_NARRATIVE_MAX_TOKENS", os.getenv("KIMI_DERMATOLOGY_NARRATIVE_MAX_TOKENS", "16384"))),
        requires_json=False,
        call_label="derm_narrative",
    )

    pathology_scores = {k: float(condition_scores[k]) for k in DERM_CLASSES}
    top_class = str(condition_scores.get("top_class", "unknown"))
    conf_score = float(condition_scores.get("confidence", 0.0) or 0.0)
    conf_label = str(condition_scores.get("confidence_label", "medium"))

    findings = _build_findings(condition_scores, critical)
    impression = _extract_impression(report_text)

    models_used: list[str] = []
    if classifier_mode == "efficientnet_b4":
        models_used.append("EfficientNet-B4-derm")
    else:
        models_used.append("openrouter-vision-derm-scores")
        if score_model_slug:
            models_used.append(score_model_slug)
    if narr_model_slug:
        models_used.append(narr_model_slug)
    else:
        models_used.append("openrouter-dermatology")

    return {
        "job_id": job_id or "",
        "modality": "dermatology",
        "status": "complete",
        "findings": findings,
        "impression": impression or report_text[:300],
        "pathology_scores": pathology_scores,
        "structures": {
            "critical": critical,
            "patient_context": patient_context,
            "narrative_report": report_text,
            "classifier_mode": classifier_mode,
            "top_condition": top_class,
        },
        "confidence": conf_label,
        "confidence_score": conf_score,
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }


def get_ready() -> dict[str, Any]:
    """Ready when OPENROUTER_API_KEY is set; V2 weights optional."""
    from config import CHECKPOINT_FILENAME, DEVICE, MODEL_DIR

    key_ok = False
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"):
        k = (os.environ.get(name) or "").strip()
        if k and len(k) >= 8:
            key_ok = True
            break
    prompt_ok = DERMAI_SYSTEM_PATH.is_file()
    wpath = Path(MODEL_DIR) / CHECKPOINT_FILENAME
    has_v2 = wpath.is_file()
    ready = key_ok and prompt_ok
    return {
        "ready": ready,
        "classifier": "loaded" if has_v2 else "openrouter_vision_v1",
        "device": DEVICE,
        "mode": "efficientnet_b4" if has_v2 else "openrouter_vision_v1",
        "openrouter_configured": key_ok,
        "dermai_prompt_present": prompt_ok,
    }


def run_dermatology_pipeline_b64(
    image_b64: str,
    patient_context: dict[str, Any] | None = None,
    job_id: str = "",
    filename_hint: str = "",
) -> dict[str, Any]:
    """ZeroClaw / agent entry; requires OPENROUTER_API_KEY."""
    _ = filename_hint

    if not get_ready().get("openrouter_configured"):
        raise RuntimeError("OPENROUTER_API_KEY is not set (see config/cloud_inference.yaml).")
    return analyze_dermatology(
        image_b64,
        patient_context or {},
        job_id=job_id or "zeroclaw",
    )
