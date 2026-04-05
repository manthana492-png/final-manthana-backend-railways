"""Manthana — Cardiac CT Inference: TotalSeg heartchambers + mask-based aorta metrics."""

import json
import logging
import os
import sys

sys.path.insert(0, "/app/shared")

from disclaimer import DISCLAIMER
from input_modality import is_mr_input
from preprocessing.ct_loader import is_degraded_single_slice, load_ct_volume
from schemas import Finding
from totalseg_runner import (
    estimate_aortic_diameter_mm,
    get_totalseg_version,
    run_totalseg,
    structure_list_from_result,
)

logger = logging.getLogger("manthana.cardiac_ct")
PIPELINE_VERSION = "manthana-ct-v2"


def _cardiac_narrative_policy() -> str:
    """
    CT_CARDIAC_NARRATIVE_POLICY:
      - off (default): no LLM narrative
      - openrouter (default) | off: OpenRouter only (SSOT: config/cloud_inference.yaml)
    """
    return os.environ.get("CT_CARDIAC_NARRATIVE_POLICY", "off").strip().lower()


def _call_cardiac_ct_narrative(
    *,
    impression: str,
    findings: list,
    pathology_scores: dict,
    patient_context: dict | None,
) -> tuple[str, list[str]]:
    policy = _cardiac_narrative_policy()
    tags: list[str] = []
    if policy in ("off", "none", "disabled", "0"):
        return "", tags

    system = (
        "You are a cardiac imaging specialist writing a concise CT narrative from structured outputs only. "
        "Do not invent measurements, stenosis grades, or calcium scores not present in the JSON."
    )
    findings_json = json.dumps(findings, indent=2)[:8000]
    scores_json = json.dumps(pathology_scores, indent=2)[:8000]
    patient_json = json.dumps(patient_context or {}, indent=2)[:4000]
    user_text = (
        f"IMPRESSION (pipeline):\n{impression}\n\n"
        f"FINDINGS:\n{findings_json}\n\n"
        f"PATHOLOGY_SCORES:\n{scores_json}\n\n"
        f"PATIENT_CONTEXT:\n{patient_json}"
    )

    try:
        from llm_router import llm_router

        out = llm_router.complete_for_role(
            "narrative_ct",
            system,
            user_text,
            max_tokens=1200,
        )
        txt = (out.get("content") or "").strip()
        if txt:
            tags.append("OpenRouter-narrative-Cardiac")
            return txt, tags
    except Exception as e:
        logger.warning("Cardiac CT OpenRouter narrative failed: %s", e)
    return "", tags


def is_loaded() -> dict:
    totalseg_ok = False
    comp2comp_ok = False
    try:
        import totalsegmentator  # noqa: F401

        totalseg_ok = True
    except ImportError:
        pass
    try:
        exe = os.path.join(os.getenv("COMP2COMP_DIR", "/opt/Comp2Comp"), "bin", "C2C")
        if os.path.isfile(exe):
            import subprocess

            r = subprocess.run(
                [exe, "--help"],
                capture_output=True,
                timeout=15,
                cwd=os.getenv("COMP2COMP_DIR", "/opt/Comp2Comp"),
            )
            comp2comp_ok = r.returncode in (0, 2)
    except Exception:
        pass
    return {"totalseg": totalseg_ok, "comp2comp": comp2comp_ok}


def run_pipeline(
    filepath: str,
    job_id: str,
    series_dir: str = "",
    source_modality: str = "",
    patient_context: dict | None = None,
) -> dict:
    logger.info(f"[{job_id}] Running cardiac CT pipeline...")
    volume, _meta, _loaded = load_ct_volume(filepath, series_dir=series_dir or None)
    degraded = is_degraded_single_slice(volume)

    tot_result: dict = {}
    try:
        inp = filepath if filepath and os.path.isfile(filepath) else volume
        tot_result = run_totalseg(
            inp,
            task="heartchambers",
            fast=True,
            device=os.getenv("TOTALSEG_DEVICE", "gpu"),
        )
    except Exception as e:
        logger.warning("TotalSegmentator cardiac task failed: %s", e)
        tot_result = {"structure_names": [], "output_dir": "", "volumes_cm3": {}}

    out_dir = tot_result.get("output_dir") or ""
    segments = structure_list_from_result(tot_result)
    volumes_cm3 = tot_result.get("volumes_cm3") or {}

    if not segments:
        segments = [
            "left_ventricle",
            "right_ventricle",
            "left_atrium",
            "right_atrium",
            "aorta",
            "pulmonary_artery",
            "pericardium",
        ]

    aorta_metrics = estimate_aortic_diameter_mm(out_dir) if out_dir else {}

    pathology_scores: dict = {}
    for k, v in volumes_cm3.items():
        pathology_scores[k] = v
    if aorta_metrics.get("max_aorta_diameter_mm") is not None:
        pathology_scores["max_aorta_diameter_mm"] = aorta_metrics.get("max_aorta_diameter_mm")
    if aorta_metrics.get("aaa_detected") is not None:
        pathology_scores["aaa_flag"] = float(bool(aorta_metrics.get("aaa_detected")))
    if aorta_metrics.get("aaa_risk_flag") is not None:
        pathology_scores["aaa_risk_flag"] = float(bool(aorta_metrics.get("aaa_risk_flag")))

    mr_study = is_mr_input(source_modality, series_dir or None)
    findings = _build_cardiac_findings(segments, aorta_metrics, pathology_scores, degraded, mr_study)
    findings_out = [f.model_dump() if isinstance(f, Finding) else f for f in findings]

    impression = "Cardiac CT analysis complete. Clinical correlation recommended."
    narrative, narr_tags = _call_cardiac_ct_narrative(
        impression=impression,
        findings=findings_out,
        pathology_scores=pathology_scores,
        patient_context=patient_context,
    )

    structures_payload = {
        "volumes_cm3": volumes_cm3,
        "segment_names": segments,
        "algorithm_version": {
            "totalsegmentator": get_totalseg_version(),
            "comp2comp_git_sha": os.getenv("COMP2COMP_GIT_SHA", "unknown"),
            "pipeline_version": PIPELINE_VERSION,
        },
        "input_type": "series" if (series_dir and os.path.isdir(series_dir)) else "single_file",
        "narrative_report": narrative,
        "narrative_policy": _cardiac_narrative_policy(),
    }
    if narrative and len(narrative) > 40:
        impression = narrative[:320].strip() + ("…" if len(narrative) > 320 else "")

    models_used = ["TotalSegmentator-heartchambers", "TotalSegmentator-AAQ-proxy"]
    models_used.extend(narr_tags)

    return {
        "modality": "cardiac_ct",
        "findings": findings_out,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures_payload,
        "confidence": "medium",
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }


def _build_cardiac_findings(
    segments: list,
    aorta: dict,
    pathology_scores: dict,
    degraded: bool,
    mr_study: bool = False,
) -> list:
    findings: list = []

    if mr_study:
        findings.append(
            Finding(
                label="MRI processed by CT-optimised pipeline",
                description=(
                    "This cardiac study was analysed with CT-optimised TotalSegmentator tasks. "
                    "If the source is MRI, results may be inaccurate — use dedicated cardiac MRI review."
                ),
                severity="warning",
                confidence=100.0,
                region="Heart",
            )
        )

    if degraded:
        findings.append(
            Finding(
                label="Degraded Input — Single Slice",
                description=(
                    "Thin or single-slice input. Cardiac chamber segmentation is most reliable "
                    "on a full CT volume."
                ),
                severity="warning",
                confidence=100.0,
                region="Heart",
            )
        )

    findings.append(
        Finding(
            label="Coronary calcium and wall assessment",
            description=(
                "Coronary artery calcium score and wall-motion or wall-thickness assessment "
                "were not performed — dedicated cardiac CT protocol and reporting are required."
            ),
            severity="info",
            confidence=100.0,
            region="Heart",
        )
    )

    aorta_mm = aorta.get("max_aorta_diameter_mm")
    if aorta_mm is not None and aorta_mm > 0:
        sev = "critical" if aorta.get("aaa_risk_flag") else ("warning" if aorta.get("aaa_detected") else "clear")
        findings.append(
            Finding(
                label="Aorta measurement (TotalSeg mask)",
                description=f"Estimated max aortic diameter {aorta_mm:.1f} mm (mask-based proxy).",
                severity=sev,
                confidence=85.0,
                region="Aorta",
            )
        )

    if segments:
        findings.append(
            Finding(
                label="Cardiac structures segmented",
                description=f"Structures: {', '.join(segments[:10])}{'...' if len(segments) > 10 else ''}",
                severity="info",
                confidence=85.0,
                region="Heart",
            )
        )

    return findings


def run_pipeline_b64(
    file_b64: str,
    patient_context: str = "",
    job_id: str = "",
    filename_hint: str = "",
    series_dir: str = "",
) -> dict:
    import base64
    import tempfile
    import uuid

    data = base64.b64decode(file_b64)
    ext = os.path.splitext(filename_hint)[1] if filename_hint else ".dcm"
    jid = job_id or str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(suffix=ext or ".dcm", delete=False) as f:
        f.write(data)
        fp = f.name
    try:
        return run_pipeline(fp, jid, series_dir=series_dir, source_modality="")
    finally:
        try:
            os.unlink(fp)
        except OSError:
            pass
