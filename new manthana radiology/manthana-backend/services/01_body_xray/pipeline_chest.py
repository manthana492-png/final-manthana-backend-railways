"""
Manthana — Chest X-Ray Pipeline
TorchXRayVision dual-model ensemble (all + chex/mimic_nb).
"""

from __future__ import annotations

import sys
import logging

sys.path.insert(0, "/app/shared")

from disclaimer import DISCLAIMER
from txrv_utils import ensemble_txrv, is_primary_loaded, get_txrv_runtime_stats
from heatmap_generator import generate_heatmap

logger = logging.getLogger("manthana.xray.chest")


def is_loaded() -> bool:
    return is_primary_loaded()


def run_chest_pipeline(filepath: str, job_id: str) -> dict:
    """Full chest X-ray analysis pipeline."""
    logger.info(f"[{job_id}] Running chest X-ray pipeline (TorchXRayVision ensemble)...")

    pathology_scores, agreement, models_used = ensemble_txrv(filepath)

    significant = [
        k
        for k, v in pathology_scores.items()
        if v is not None and v > 0.5 and k != "no_finding"
    ]

    impression = _build_impression(significant, agreement)
    confidence = _confidence_str(agreement)

    structured_findings = _build_chest_findings(pathology_scores, significant)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "narrative (local only):\n%s",
            "\n".join(_build_narrative(pathology_scores, agreement, models_used)),
        )
    findings_for_heatmap = structured_findings

    # heatmap_url — AnalysisResponse schema (NOT heatmap_path)
    heatmap_url = generate_heatmap(
        image_path=filepath,
        job_id=job_id,
        findings=findings_for_heatmap,
        pathology_scores=pathology_scores,
        detected_region="chest",
        model=None,
        target_layer=None,
        input_tensor=None,
    )

    over_30 = {k: float(v) for k, v in pathology_scores.items() if v > 0.3}
    heatmap_type = "none" if heatmap_url is None else "synthetic"
    structures: dict = {
        "pathology_fractions": over_30,
        "display_lines": [f"{k}: {v:.1%}" for k, v in pathology_scores.items() if v > 0.3],
        "narrative_report": "",
        "heatmap_type": heatmap_type,
        "runtime": get_txrv_runtime_stats(),
    }

    return {
        "modality": "xray",
        "detected_region": "chest",
        "findings": structured_findings,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": confidence,
        "heatmap_url": heatmap_url,
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }


def _confidence_str(agreement: float) -> str:
    if agreement > 0.75:
        return "high"
    if agreement > 0.5:
        return "medium"
    return "low"


def _build_impression(significant: list[str], agreement: float) -> str:
    _ = agreement
    if not significant:
        return "No significant abnormality detected on chest X-ray."
    return (
        f"Findings suggestive of: {', '.join(significant)}. "
        "Clinical correlation recommended."
    )


def _build_narrative(
    pathology_scores: dict,
    agreement: float,
    models_used: list[str],
) -> list[str]:
    """Local only — heatmap / internal; not returned on AnalysisResponse."""
    top = sorted(
        pathology_scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:8]
    lines = [
        "Chest X-ray automated interpretation (TorchXRayVision ensemble).",
        f"Models: {', '.join(models_used)}.",
        f"Ensemble agreement index: {agreement:.2f}.",
        "",
        "Key predicted findings:",
    ]
    for name, sc in top:
        if sc > 0.2:
            lines.append(f"  • {name}: {sc:.1%}")
    lines.append("")
    lines.append(
        "Clinical correlation and review of the source imaging are required before any clinical decision."
    )
    return lines


def _build_chest_findings(scores: dict, significant: list[str]) -> list:
    """Build structured Finding dicts; order significant paths first when present."""
    sig_set = set(significant)

    def sort_key(kv: tuple[str, float]) -> tuple[int, float]:
        name, sc = kv
        return (0 if name in sig_set else 1, -sc)

    findings = []

    for name, score in sorted(scores.items(), key=sort_key):
        if score <= 0.1 or name == "no_finding":
            continue

        if score > 0.7:
            severity = "critical"
        elif score > 0.5:
            severity = "warning"
        elif score > 0.3:
            severity = "info"
        else:
            severity = "clear"

        findings.append(
            {
                "label": name.replace("_", " ").title(),
                "severity": severity,
                "confidence": round(score * 100, 1),
                "region": "Chest",
                "description": f"{name.replace('_', ' ').title()} detected with {score:.0%} probability",
            }
        )

    if not findings:
        findings.append(
            {
                "label": "No significant abnormality",
                "severity": "clear",
                "confidence": 95.0,
                "region": "Chest",
                "description": "No significant pathology detected on automated screening.",
            }
        )

    return findings
