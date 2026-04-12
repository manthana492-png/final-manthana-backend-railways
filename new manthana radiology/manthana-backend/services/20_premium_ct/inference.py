from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.normpath(os.path.join(_ROOT, "..", "..", "shared")), "/app/shared"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from disclaimer import DISCLAIMER
from preprocessing.ct_loader import is_degraded_single_slice, load_ct_volume
from schemas import Finding
from segmentation_export import export_segmentation_nifti, voxel_spacing_from_meta
from vista3d_full_integration import run_vista3d_segmentation

import config as premium_config

logger = logging.getLogger("manthana.premium_ct")
PIPELINE_VERSION = "manthana-premium-ct-v1"


def is_loaded() -> dict[str, Any]:
    resolved = premium_config.resolve_vista3d_checkpoint_path()
    checkpoints_present = os.path.isfile(resolved)
    return {
        "vista3d_enabled": premium_config.VISTA3D_ENABLED,
        "vista3d_full_forward": premium_config.VISTA3D_FULL_FORWARD,
        "vista3d_checkpoint_present": checkpoints_present,
        "vista3d_checkpoint_path": resolved,
        "vista3d_model_path_env": premium_config.VISTA3D_MODEL_PATH,
    }


def _prompt_for_premium_ct(vista_results: dict[str, Any], patient_context: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are a senior radiology assistant. Build a concise but structured premium CT report "
        "from VISTA-3D volumetric segmentation outputs only. Do not invent findings."
    )
    user_text = (
        f"REGION: {vista_results.get('region_analyzed')}\n"
        f"CLASSES_DETECTED: {vista_results.get('classes_detected')}\n"
        f"VOLUMES_ML:\n{json.dumps(vista_results.get('volumes_ml') or {}, indent=2)[:10000]}\n\n"
        f"CLASS_SCORES:\n{json.dumps(vista_results.get('class_scores') or {}, indent=2)[:10000]}\n\n"
        f"PATIENT_CONTEXT:\n{json.dumps(patient_context or {}, indent=2)[:4000]}\n\n"
        "Return report sections: Technique, Findings by body system, Measurements, Impression, Recommendations."
    )
    return system, user_text


def _build_findings(vista_results: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    top = sorted(
        (vista_results.get("class_scores") or {}).items(),
        key=lambda kv: float(kv[1]),
        reverse=True,
    )[:8]
    for name, score in top:
        findings.append(
            Finding(
                label=f"VISTA-3D: {name}",
                severity="info",
                confidence=round(float(score) * 100.0, 2),
                description="Detected by full-volume 127-class VISTA segmentation.",
                region=vista_results.get("region_analyzed") or "full_body",
            )
        )
    if not findings:
        findings.append(
            Finding(
                label="No high-confidence segmented structures",
                severity="warning",
                confidence=0.0,
                description="Premium CT segmentation ran but no classes passed confidence threshold.",
                region=vista_results.get("region_analyzed") or "full_body",
            )
        )
    return findings


def run_pipeline(
    filepath: str,
    job_id: str,
    *,
    series_dir: str = "",
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    patient_context = patient_context or {}
    premium_context = patient_context.get("premium_ct") if isinstance(patient_context, dict) else {}
    region_hint = "full_body"
    if isinstance(premium_context, dict):
        region_hint = str(premium_context.get("vista_region_preference") or "full_body")

    volume, meta, _series_available = load_ct_volume(filepath, series_dir=series_dir or None)
    if is_degraded_single_slice(np.asarray(volume)):
        raise ValueError(
            "Premium 3D CT requires volumetric input. Upload DICOM series ZIP or NIfTI volume."
        )
    spacing_xyz = voxel_spacing_from_meta(meta if isinstance(meta, dict) else {})

    vista_results = run_vista3d_segmentation(
        volume=np.asarray(volume),
        spacing_xyz=spacing_xyz,
        model_path=premium_config.resolve_vista3d_checkpoint_path(),
        device=premium_config.DEVICE,
        region_hint=region_hint,
    )
    seg_mask = vista_results.get("segmentation_mask")
    seg_path = ""
    if isinstance(seg_mask, np.ndarray):
        out_dir = Path("/tmp/manthana_premium_ct")
        seg_path = str(
            export_segmentation_nifti(seg_mask, out_dir / f"{job_id}_vista3d_mask.nii.gz")
        )

    narrative = ""
    models_used = ["NVIDIA VISTA-3D", "MONAI"]
    try:
        from llm_router import llm_router

        system, user_text = _prompt_for_premium_ct(vista_results, patient_context)
        llm = llm_router.complete_for_role(
            "narrative_premium_ct",
            system,
            user_text,
            max_tokens=3200,
        )
        narrative = (llm.get("content") or "").strip()
        if narrative:
            models_used.append("Kimi K2.5 (OpenRouter)")
    except Exception as e:  # noqa: BLE001
        logger.warning("Premium CT narrative fallback (LLM unavailable): %s", e)

    findings = _build_findings(vista_results)
    impression = (
        narrative.split("\n", 1)[0][:280]
        if narrative
        else f"Premium VISTA-3D segmentation completed ({vista_results.get('classes_detected', 0)} classes detected)."
    )

    return {
        "job_id": job_id,
        "status": "complete",
        "modality": premium_config.SERVICE_NAME,
        "findings": findings,
        "impression": impression,
        "pathology_scores": {
            "vista3d": {
                "classes_detected": vista_results.get("classes_detected"),
                "class_scores": vista_results.get("class_scores"),
                "volumes_ml": vista_results.get("volumes_ml"),
                "region_analyzed": vista_results.get("region_analyzed"),
            }
        },
        "structures": {
            "segmentation_mask_path": seg_path,
            "vista3d_region_preference": region_hint,
            "voxel_spacing_mm": spacing_xyz,
            "processing_steps": [
                "upload_validate",
                "vista_segmentation",
                "volume_measurements",
                "narrative_generation",
            ],
            "narrative": narrative,
        },
        "confidence": "high",
        "heatmap_url": None,
        "processing_time_sec": round(time.time() - t0, 2),
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
        "detected_region": region_hint,
        "analysis_depth": "deep",
    }

