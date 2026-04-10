"""
Manthana — ECG Inference Pipeline
Heuristic rhythm scoring (ecg_rhythm) + neurokit2 intervals; narrative via OpenRouter narrative_ecg (SSOT prompts).
No downloadable model weights — no Modal volume checkpoints for this service.
"""

import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

_ecg_dir = Path(__file__).resolve().parent
_backend_shared = _ecg_dir.parent.parent / "shared"
if _backend_shared.is_dir():
    sys.path.insert(0, str(_backend_shared))
sys.path.insert(0, "/app/shared")

from disclaimer import DISCLAIMER
from ecg_intervals import compute_ecg_intervals
from ecg_rhythm import (
    RHYTHM_KEYS,
    ensemble_rhythm_dict,
    rhythm_scores_from_signal,
    rhythm_scores_secondary,
)
from preprocessing.ecg_utils import (
    detect_input_type,
    read_ecg_csv,
    read_ecg_edf,
    read_ecg_dicom,
    normalize_ecg,
)
from schemas import Finding

from config import (
    ECG_IMAGE_BLUR_VARIANCE_MIN,
    ECG_IMAGE_MIN_SHORT_EDGE,
    ECG_IMAGE_QUALITY_GATE,
    ECG_PIPELINE_VERSION,
)
from digitiser_adapter import digitize_ecg_image_adapted
from ecg_image_quality import assess_ecg_image_quality
from patient_context_parser import format_ecg_patient_prompt_section

logger = logging.getLogger("manthana.ecg")

# Rhythm keys that indicate normal physiology — high score is good, not abnormality
_EXCLUDED_FROM_ABNORMAL = frozenset({"sinus_rhythm", "normal_rhythm"})

ENGINE_LABEL = "Manthana-ECG-Engine"
_ECG_PROMPT_DEFAULT = Path(__file__).resolve().parent / "prompts" / "ecg_system.md"

_EMERGENCY_FINDING_COPY: dict[str, tuple[str, str]] = {
    "stemi": (
        "STEMI — emergency protocol",
        "EMERGENCY — Activate cath lab / thrombolysis within 90 minutes. STEMI protocol NOW.",
    ),
    "long_qt": (
        "Long QT (QTc prolonged) — urgent",
        "URGENT — Stop all QT-prolonging drugs. Check electrolytes. Cardiology review today.",
    ),
    "ventricular_tachycardia": (
        "Ventricular tachycardia — emergency",
        "EMERGENCY — Resuscitation team immediately. Do not leave patient alone.",
    ),
    "ventricular_fibrillation": (
        "Ventricular fibrillation — emergency",
        "EMERGENCY — Resuscitation team immediately. Do not leave patient alone.",
    ),
    "complete_heart_block": (
        "Complete heart block — urgent",
        "URGENT — Cardiology consult immediately. Temporary pacing may be required.",
    ),
    "afib_rvr": (
        "Atrial fibrillation with rapid ventricular response",
        "URGENT — Rate control required. Stroke risk assessment (CHA₂DS₂-VASc).",
    ),
}


def is_loaded() -> bool:
    """Pure CPU signal pipeline — always available after import."""
    return True


def get_ecg_pipeline_status() -> dict[str, Any]:
    """Health / debug: pipeline id (no DL weights)."""
    return {
        "ecg_pipeline_version": ECG_PIPELINE_VERSION,
        "ecg_branch": "heuristic_neurokit2_openrouter",
        "ecg_dl_weights": "none",
    }


def _digitizer_available() -> bool:
    try:
        from digitizer import digitize_ecg_image  # noqa: F401

        return True
    except ImportError:
        return False


def _score_to_severity(score: float) -> str:
    if score > 0.75:
        return "critical"
    if score > 0.60:
        return "warning"
    if score > 0.45:
        return "info"
    return "clear"


def _score_to_confidence(score: float) -> float:
    return round(float(score) * 100.0, 1)


def _significant_abnormalities(rhythm_scores: dict[str, float]) -> list[str]:
    return [
        k
        for k, v in rhythm_scores.items()
        if isinstance(v, (int, float)) and float(v) > 0.5 and k not in _EXCLUDED_FROM_ABNORMAL
    ]


def _build_ecg_findings(
    rhythm_scores: dict[str, float],
    significant: list[str],
) -> list[Finding]:
    if not significant:
        return [
            Finding(
                label="No Significant Rhythm Abnormality",
                description="All rhythm scores within normal screening range.",
                severity="clear",
                confidence=95.0,
                region="cardiac",
            )
        ]

    findings: list[Finding] = []
    for name in significant:
        score = float(rhythm_scores.get(name, 0.0) or 0.0)
        findings.append(
            Finding(
                label=name.replace("_", " ").title(),
                description=f"Rhythm abnormality probability: {score:.0%}",
                severity=_score_to_severity(score),
                confidence=_score_to_confidence(score),
                region="cardiac",
            )
        )
    return findings


def _read_ecg_system_prompt(prompt_path: Optional[Path]) -> str:
    p = prompt_path or _ECG_PROMPT_DEFAULT
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8")[:20000]
    except OSError:
        pass
    return (
        "You are a senior cardiologist. Interpret ECG findings conservatively from supplied metrics only."
    )


def _top_3_rhythm_line(rhythm_scores: dict[str, Any]) -> str:
    nums = [
        (k, float(rhythm_scores[k]))
        for k in RHYTHM_KEYS
        if k in rhythm_scores and isinstance(rhythm_scores[k], (int, float))
    ]
    nums.sort(key=lambda x: -x[1])
    return ", ".join(f"{k}={v:.3f}" for k, v in nums[:3])


def _build_ecg_text_payload(
    rhythm_scores: dict[str, Any],
    intervals: dict[str, Any],
    emergency_flags: list[str],
    patient_context: Optional[dict],
    structures: Optional[dict],
) -> str:
    ctx = patient_context if isinstance(patient_context, dict) else {}
    st = structures or {}
    age = ctx.get("age", "")
    sex = ctx.get("sex", "")
    clin = ctx.get("clinical_history", ctx.get("history", ""))
    structured_block = format_ecg_patient_prompt_section(ctx)
    payload = f"""ECG Analysis Results:
Rhythm: {st.get('rhythm')}
Heart Rate: {st.get('heart_rate_bpm')} bpm
PR interval: {intervals.get('pr_ms')} ms
QRS duration: {intervals.get('qrs_ms')} ms
QTc: {intervals.get('qtc_ms')} ms
Emergency flags: {emergency_flags}

Top rhythm scores: {_top_3_rhythm_line(rhythm_scores)}

Patient summary: {age}y {sex}, {clin}

Generate ECG report per system prompt format."""
    if structured_block:
        payload += "\n\n" + structured_block
    if st.get("input_type") == "image":
        payload += (
            "\n\nNote: QTc estimated from digitised image. "
            "Accuracy ±30ms. Confirm with direct waveform "
            "measurement before clinical action on QTc."
        )
    return payload


def _call_ecg_narrative(
    rhythm_scores: dict[str, Any],
    intervals: dict[str, Any],
    emergency_flags: list[str],
    patient_context: Optional[dict],
    image_b64: Optional[str] = None,
    structures: Optional[dict] = None,
    prompt_path: Optional[Path] = None,
) -> tuple[str, list[str]]:
    """OpenRouter: narrative_ecg role (vision if digitized image, else text). Never raises."""
    tags: list[str] = []
    system = _read_ecg_system_prompt(prompt_path)
    text_payload = _build_ecg_text_payload(
        rhythm_scores, intervals, emergency_flags, patient_context, structures
    )

    try:
        from llm_router import llm_router
    except Exception as exc:
        logger.warning("ECG narrative: llm_router unavailable: %s", exc)
        return "", []

    if image_b64:
        try:
            from PIL import Image as _PIL

            raw = base64.b64decode(image_b64)
            img = _PIL.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            jpeg_b64 = base64.b64encode(buf.getvalue()).decode()
            out = llm_router.complete_for_role(
                "narrative_ecg",
                system[:20000],
                text_payload[:120000],
                image_b64=jpeg_b64,
                image_mime="image/jpeg",
                max_tokens=1200,
                temperature=0.2,
            )
            txt = (out.get("content") or "").strip()
            if txt:
                tags.append("OpenRouter-vision-ECG")
                return txt, tags
        except Exception as e:
            logger.warning("OpenRouter vision ECG failed: %s", e)

    try:
        out = llm_router.complete_for_role(
            "narrative_ecg",
            system[:20000],
            text_payload[:120000],
            max_tokens=1200,
            temperature=0.2,
        )
        txt = (out.get("content") or "").strip()
        if txt:
            tags.append("OpenRouter-text-ECG")
            return txt, tags
    except Exception as e:
        logger.warning("OpenRouter text ECG failed: %s", e)

    return "", []


def _enrich_structures(
    parameters: dict,
    rhythm_scores: dict[str, float],
    significant: list[str],
    input_type: str,
) -> dict:
    """Merge interval dict with report/correlation fields; narrative filled later."""
    if significant:
        rhythm = significant[0]
    else:
        sr = float(rhythm_scores.get("sinus_rhythm", 0.0) or 0.0)
        rhythm = "sinus_rhythm" if sr > 0.5 else "no_significant_abnormality"

    intervals = {
        "pr_ms": parameters.get("pr_ms"),
        "qrs_ms": parameters.get("qrs_ms"),
        "qt_ms": parameters.get("qt_ms"),
        "qtc_ms": parameters.get("qtc_ms"),
    }
    hr_raw = parameters.get("hr_bpm")
    hr_bpm = float(hr_raw) if hr_raw is not None else 0.0
    pr_ms = parameters.get("pr_ms")
    qrs_ms = parameters.get("qrs_ms")
    qtc = parameters.get("qtc_ms")
    method = str(parameters.get("method", "")).lower()
    interval_reliable = not (input_type == "image" and "fallback" in method)

    emergency_flags: list[str] = []
    if float(rhythm_scores.get("st_elevation", 0.0) or 0.0) > 0.5:
        emergency_flags.append("stemi")
    if interval_reliable and qtc is not None and float(qtc) > 500:
        emergency_flags.append("long_qt")

    rs = {k: float(v) for k, v in rhythm_scores.items() if isinstance(v, (int, float))}
    if rs.get("ventricular_tachycardia", 0.0) > 0.5:
        emergency_flags.append("ventricular_tachycardia")
    elif hr_bpm > 120 and qrs_ms is not None and float(qrs_ms) > 120:
        emergency_flags.append("ventricular_tachycardia")

    if rs.get("ventricular_fibrillation", 0.0) > 0.5:
        emergency_flags.append("ventricular_fibrillation")

    if rs.get("complete_heart_block", 0.0) > 0.5:
        emergency_flags.append("complete_heart_block")
    elif (
        interval_reliable
        and
        pr_ms is not None
        and float(pr_ms) > 300
        and hr_bpm > 0
        and hr_bpm < 45
    ):
        emergency_flags.append("complete_heart_block")

    if rs.get("atrial_fibrillation", 0.0) > 0.6 and hr_bpm > 150:
        emergency_flags.append("afib_rvr")

    # de-dupe preserve order
    seen: set[str] = set()
    emergency_flags = [x for x in emergency_flags if not (x in seen or seen.add(x))]

    out = dict(parameters)
    out["rhythm"] = rhythm
    out["heart_rate_bpm"] = parameters.get("hr_bpm")
    out["intervals"] = intervals
    out["narrative_report"] = ""
    out["emergency_flags"] = emergency_flags
    out["interval_reliable"] = interval_reliable
    out["input_type"] = input_type
    out["india_note"] = (
        "India context: high hypertension prevalence; young STEMI not uncommon; "
        "AF in young adults — consider RHD (mitral) screening when clinically appropriate."
    )
    return out


def _append_emergency_findings(emergency_flags: list[str]) -> list[Finding]:
    """One critical Finding per emergency flag (standardised copy)."""
    out: list[Finding] = []
    for flag in emergency_flags:
        pair = _EMERGENCY_FINDING_COPY.get(flag)
        if not pair:
            continue
        label, desc = pair
        out.append(
            Finding(
                label=label,
                description=desc,
                severity="critical",
                confidence=95.0,
                region="cardiac",
            )
        )
    return out


def _build_impression(
    rhythm_scores: dict[str, float],
    parameters: dict,
    significant: list[str],
) -> str:
    hr = parameters.get("hr_bpm")
    qtc = parameters.get("qtc_ms")
    hr_str = f"HR {hr:.0f} bpm. " if hr is not None else ""

    if not significant:
        return f"{hr_str}Normal sinus rhythm. No significant rhythm abnormality detected."

    top = significant[0].replace("_", " ").title()
    score = float(rhythm_scores.get(significant[0], 0.0) or 0.0)
    qtc_flag = ""
    if qtc is not None and qtc > 450:
        qtc_flag = f" QTc prolonged at {qtc:.0f} ms."
    return f"{hr_str}{top} detected (p={score:.2f}).{qtc_flag} Clinical correlation recommended."


def _pathology_scores_for_response(
    rhythm_scores: dict[str, float],
    structures: dict,
    patient_context: Optional[dict],
) -> dict[str, float]:
    """Rhythm scores plus correlation keys (qtc_ms, afib_confidence, patient_age)."""
    out: dict[str, float] = {
        k: float(v) for k, v in rhythm_scores.items() if isinstance(v, (int, float))
    }
    iv = structures.get("intervals") or {}
    if iv.get("qtc_ms") is not None:
        try:
            out["qtc_ms"] = float(iv["qtc_ms"])
        except (TypeError, ValueError):
            pass
    out["afib_confidence"] = float(out.get("atrial_fibrillation", 0.0) or 0.0)
    ctx = patient_context or {}
    age = ctx.get("age")
    if age is not None:
        try:
            out["patient_age"] = float(age)
        except (TypeError, ValueError):
            pass
    return out


def run_ecg_pipeline(
    filepath: str,
    job_id: str,
    patient_context: Optional[dict] = None,
    image_b64: Optional[str] = None,
) -> dict:
    input_type = detect_input_type(filepath)
    logger.info(f"[{job_id}] ECG input type: {input_type}")

    if input_type == "pdf_rejected":
        raise ValueError(
            "PDF ECG files are not supported. Upload a digital ECG (.csv, .edf, .dcm) "
            "or a clear photo (.jpg, .png) of the printout."
        )

    models_used: list[str] = []
    timing_ms: dict[str, float] = {}
    image_quality: dict[str, Any] = {}

    if input_type == "image":
        if ECG_IMAGE_QUALITY_GATE:
            image_quality = assess_ecg_image_quality(
                filepath,
                min_short_edge=ECG_IMAGE_MIN_SHORT_EDGE,
                blur_variance_min=ECG_IMAGE_BLUR_VARIANCE_MIN,
            )
        logger.info(f"[{job_id}] Digitizing ECG photo...")
        t0 = time.perf_counter()
        signal, sample_rate = _digitize_ecg_photo(filepath)
        timing_ms["digitize_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        ext_used = os.getenv("ECG_DIGITISER_REPO_ROOT", "").strip()
        if ext_used and os.path.isdir(ext_used):
            models_used.append("ecg-digitiser-external")
        elif _digitizer_available():
            models_used.append("ecg-digitiser-opencv")
        else:
            models_used.append("ecg-signal-extractor")
            logger.warning(
                "ECG digitizer module unavailable — simplified CV extraction used."
            )
    elif input_type == "csv":
        signal, sample_rate = read_ecg_csv(filepath)
    elif input_type == "edf":
        signal, sample_rate = read_ecg_edf(filepath)
    elif input_type == "dicom":
        signal, sample_rate = read_ecg_dicom(filepath)
    else:
        raise ValueError(
            f"Unsupported ECG input: {input_type}. "
            "Supported: JPEG/PNG photo, CSV, EDF, DICOM-ECG"
        )

    t_norm = time.perf_counter()
    signal = normalize_ecg(signal, target_rate=500, current_rate=sample_rate)
    sample_rate = 500.0
    timing_ms["normalize_ms"] = round((time.perf_counter() - t_norm) * 1000.0, 2)

    fm_scores = rhythm_scores_from_signal(signal, 500.0)
    hl_scores = rhythm_scores_secondary(signal, 500.0)
    rhythm_scores = ensemble_rhythm_dict(fm_scores, hl_scores)
    models_used.append(ENGINE_LABEL)

    parameters = compute_ecg_intervals(signal, sample_rate=500.0)
    significant = _significant_abnormalities(rhythm_scores)
    impression = _build_impression(rhythm_scores, parameters, significant)
    structures = _enrich_structures(parameters, rhythm_scores, significant, input_type=input_type)
    structures = dict(structures)
    structures["ecg_pipeline_version"] = ECG_PIPELINE_VERSION
    structures["ecg_timing_ms"] = timing_ms
    structures["ecg_founder_scores"] = {}
    structures["ecg_image_quality"] = image_quality
    if image_quality.get("warnings"):
        structures["ecg_image_quality_warnings"] = image_quality["warnings"]
    if parameters.get("method") == "neurokit2" and "neurokit2" not in models_used:
        models_used.append("neurokit2")

    findings = _build_ecg_findings(rhythm_scores, significant)
    findings = list(findings) + _append_emergency_findings(
        structures.get("emergency_flags") or []
    )

    pathology_scores = _pathology_scores_for_response(
        rhythm_scores, structures, patient_context
    )

    narr_scores = dict(pathology_scores)
    narrative, narr_tags = _call_ecg_narrative(
        rhythm_scores=narr_scores,
        intervals=structures.get("intervals") or {},
        emergency_flags=structures.get("emergency_flags") or [],
        patient_context=patient_context,
        image_b64=image_b64,
        structures=structures,
        prompt_path=None,
    )
    structures["narrative_report"] = narrative
    for tag in narr_tags:
        if tag not in models_used:
            models_used.append(tag)

    return {
        "modality": "ecg",
        "findings": findings,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": _assess_confidence(rhythm_scores),
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }


def _digitize_ecg_photo(filepath: str) -> tuple:
    try:
        return digitize_ecg_image_adapted(filepath, target_rate=500)
    except ImportError:
        logger.warning("ECG Digitiser not available. Using simplified extraction.")
        return _simplified_ecg_extract(filepath)


def _simplified_ecg_extract(filepath: str) -> tuple:
    import cv2

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {filepath}")
    h, w = img.shape
    signal = np.zeros((12, w), dtype=np.float32)
    for col in range(w):
        column = img[:, col]
        darkest = int(np.argmin(column))
        signal[1, col] = (h / 2 - darkest) / (h / 2)
    signal[0] = signal[1] * 0.7
    signal[2] = signal[1] * 0.3
    return signal, 500


def _assess_confidence(scores: dict) -> str:
    nums = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not nums:
        return "low"
    mx = max(nums)
    if mx > 0.75:
        return "high"
    if mx > 0.45:
        return "medium"
    return "low"


def run_ecg_pipeline_b64(
    image_b64: str,
    patient_context: str = "",
    job_id: str = "",
    filename_hint: str = "",
) -> dict:
    """
    Entry point for agent/ZeroClaw: decode base64 to a temp file and run the pipeline.
    patient_context: JSON string with age, sex, clinical_history for narrative/correlation.
    """
    pc: dict = {}
    if patient_context and str(patient_context).strip():
        try:
            loaded = json.loads(patient_context)
            if isinstance(loaded, dict):
                pc = loaded
        except (json.JSONDecodeError, TypeError, ValueError):
            pc = {}
    img_bytes = base64.b64decode(image_b64)
    if filename_hint:
        ext = os.path.splitext(filename_hint)[1].lower() or ".png"
    else:
        ext = ".png"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(img_bytes)
        filepath = f.name
    try:
        return run_ecg_pipeline(
            filepath,
            job_id or "zeroclaw",
            patient_context=pc,
            image_b64=image_b64,
        )
    finally:
        os.unlink(filepath)
