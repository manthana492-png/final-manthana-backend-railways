"""
Manthana Dermatology Engine — Kimi K2.5 vision (DermAI system prompt) + structured scores + narrative.
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
import threading
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
_openai_client_lock = threading.Lock()
_openai_client_sig: tuple[Any, ...] | None = None
_openai_client: Any = None


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


def _make_openai_client(
    api_key: str,
    base_url: str,
    *,
    timeout_sec: float,
    max_retries: int,
) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_sec,
        max_retries=max_retries,
    )


def _get_openai_client() -> Any:
    """Thread-safe singleton; recreates client when key/base_url/timeout/retry change."""
    global _openai_client_sig, _openai_client
    from config import (
        KIMI_API_KEY,
        KIMI_BASE_URL,
        KIMI_DERMATOLOGY_MAX_RETRIES,
        KIMI_DERMATOLOGY_TIMEOUT_SEC,
    )

    api_key = (KIMI_API_KEY or "").strip()
    base_url = (KIMI_BASE_URL or "https://api.moonshot.ai/v1").strip()
    sig = (
        api_key,
        base_url,
        float(KIMI_DERMATOLOGY_TIMEOUT_SEC),
        int(KIMI_DERMATOLOGY_MAX_RETRIES),
    )
    with _openai_client_lock:
        if _openai_client is None or _openai_client_sig != sig:
            _openai_client = _make_openai_client(
                api_key,
                base_url,
                timeout_sec=float(KIMI_DERMATOLOGY_TIMEOUT_SEC),
                max_retries=int(KIMI_DERMATOLOGY_MAX_RETRIES),
            )
            _openai_client_sig = sig
        return _openai_client


def _load_dermai_system_prompt() -> str:
    global _dermai_system_text
    if _dermai_system_text is None:
        _dermai_system_text = DERMAI_SYSTEM_PATH.read_text(encoding="utf-8")
    return _dermai_system_text


def _kimi_extra_body(model: str) -> dict | None:
    """Moonshot kimi-k2.5: thinking on/off only (see platform.moonshot.ai K2.5 docs)."""
    m = (model or "").lower()
    if "kimi-k2" not in m:
        return None
    from config import KIMI_DERMATOLOGY_THINKING

    if KIMI_DERMATOLOGY_THINKING == "disabled":
        return {"thinking": {"type": "disabled"}}
    return {"thinking": {"type": "enabled"}}


def _sniff_media_type(image_bytes: bytes) -> str:
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"


def _jpeg_data_url_from_bytes(image_bytes: bytes) -> str:
    """Normalize to JPEG data URL for Kimi vision (matches lab_report pattern)."""
    raw = image_bytes
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpeg_b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{jpeg_b64}"
    except Exception:
        b64 = base64.b64encode(raw).decode()
        mt = _sniff_media_type(raw)
        return f"data:{mt};base64,{b64}"


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


def _kimi_chat_completion(
    *,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    max_tokens: int,
    call_label: str = "kimi",
) -> str:
    from config import KIMI_API_KEY, KIMI_DERMATOLOGY_MODEL

    api_key = (KIMI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("KIMI_API_KEY or MOONSHOT_API_KEY is not set")

    model = (KIMI_DERMATOLOGY_MODEL or "kimi-k2.5").strip()
    extra = _kimi_extra_body(model)
    client = _get_openai_client()

    create_kw: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
    }
    if extra is not None:
        create_kw["extra_body"] = extra

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**create_kw)
    except Exception as e:
        logger.exception("%s: Kimi API request failed: %s", call_label, e)
        raise
    elapsed = time.perf_counter() - t0
    logger.debug("%s: Kimi chat.completions done in %.2fs", call_label, elapsed)

    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if hasattr(msg, "reasoning_content"):
        rc = getattr(msg, "reasoning_content", None)
        if rc and not text:
            logger.warning(
                "%s: Kimi returned empty content; reasoning present only (truncation?)",
                call_label,
            )
    return text


def _get_kimi_condition_scores(
    patient_context: dict[str, Any],
    system_prompt: str,
    image_data_url: str,
) -> dict[str, Any]:
    """Kimi K2.5 vision — JSON condition scores for pipeline compatibility."""
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
    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": score_user},
    ]
    text = _kimi_chat_completion(
        system_prompt=system_prompt,
        user_content=user_content,
        max_tokens=int(os.getenv("KIMI_DERMATOLOGY_SCORE_MAX_TOKENS", "8192")),
        call_label="derm_scores",
    )
    raw = _parse_json_from_llm(text)
    return _normalize_condition_scores(raw)


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
    Two Kimi calls when scores come from vision; one Kimi call if V2 classifier supplies scores.
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
    image_data_url = _jpeg_data_url_from_bytes(image_bytes)

    if _derm_classifier is not None:
        condition_scores = _derm_classifier.classify(pil_image)
        classifier_mode = "efficientnet_b4"
    else:
        condition_scores = _get_kimi_condition_scores(
            patient_context, system_prompt, image_data_url
        )
        classifier_mode = "kimi_k2.5_vision_v1"

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

    narrative_user: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {
            "type": "text",
            "text": (
                f"QUANTITATIVE MEASUREMENTS:\n{measurements_json}\n\n"
                "Generate the dermatology assessment report following your instructions "
                "(morphology, differential, India context, safety). "
                "If your instructions include a JSON structured block for narrative output, you may use it; "
                "otherwise use clear section headings including IMPRESSION."
            ),
        },
    ]
    report_text = _kimi_chat_completion(
        system_prompt=system_prompt,
        user_content=narrative_user,
        max_tokens=int(os.getenv("KIMI_DERMATOLOGY_NARRATIVE_MAX_TOKENS", "16384")),
        call_label="derm_narrative",
    )

    pathology_scores = {k: float(condition_scores[k]) for k in DERM_CLASSES}
    top_class = str(condition_scores.get("top_class", "unknown"))
    conf_score = float(condition_scores.get("confidence", 0.0) or 0.0)
    conf_label = str(condition_scores.get("confidence_label", "medium"))

    findings = _build_findings(condition_scores, critical)
    impression = _extract_impression(report_text)

    from config import KIMI_DERMATOLOGY_MODEL

    kimi_model = KIMI_DERMATOLOGY_MODEL or "kimi-k2.5"
    models_used: list[str] = []
    if classifier_mode == "efficientnet_b4":
        models_used.append("EfficientNet-B4-derm")
    else:
        models_used.append("kimi-k2.5-vision-derm-scores")
    models_used.append(str(kimi_model))

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
    """Ready when Kimi (Moonshot) API key is set; V2 weights optional."""
    from config import (
        CHECKPOINT_FILENAME,
        DEVICE,
        KIMI_API_KEY,
        KIMI_BASE_URL,
        KIMI_DERMATOLOGY_MODEL,
        KIMI_DERMATOLOGY_MAX_RETRIES,
        KIMI_DERMATOLOGY_TIMEOUT_SEC,
        MODEL_DIR,
    )

    key_ok = bool((KIMI_API_KEY or "").strip())
    prompt_ok = DERMAI_SYSTEM_PATH.is_file()
    wpath = Path(MODEL_DIR) / CHECKPOINT_FILENAME
    has_v2 = wpath.is_file()
    ready = key_ok and prompt_ok
    return {
        "ready": ready,
        "classifier": "loaded" if has_v2 else "kimi_k2.5_vision_v1",
        "device": DEVICE,
        "mode": "efficientnet_b4" if has_v2 else "kimi_k2.5_vision_v1",
        "kimi_configured": key_ok,
        "dermai_prompt_present": prompt_ok,
        "kimi_model": KIMI_DERMATOLOGY_MODEL,
        "kimi_base_url": KIMI_BASE_URL,
        "kimi_timeout_sec": KIMI_DERMATOLOGY_TIMEOUT_SEC,
        "kimi_max_retries": KIMI_DERMATOLOGY_MAX_RETRIES,
    }


def run_dermatology_pipeline_b64(
    image_b64: str,
    patient_context: dict[str, Any] | None = None,
    job_id: str = "",
    filename_hint: str = "",
) -> dict[str, Any]:
    """ZeroClaw / agent entry; requires KIMI_API_KEY or MOONSHOT_API_KEY."""
    _ = filename_hint
    from config import KIMI_API_KEY

    if not (KIMI_API_KEY or "").strip():
        raise RuntimeError("KIMI_API_KEY or MOONSHOT_API_KEY is not set")
    return analyze_dermatology(
        image_b64,
        patient_context or {},
        job_id=job_id or "zeroclaw",
    )
