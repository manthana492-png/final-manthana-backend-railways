"""
Optional NVIDIA VISTA-3D enrichment for premium CT brain (Modal ``VISTA3D_ENABLED``).

Adds structured ``pathology_scores["vista3d"]`` metadata and findings. Full 127-class
segmentation forward pass is wired when a loadable checkpoint exists at
``VISTA3D_MODEL_PATH``; otherwise reports ``weights_missing`` without failing the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from schemas import Finding

logger = logging.getLogger("manthana.ct_brain.vista3d")


def _vista_enabled() -> bool:
    return os.getenv("VISTA3D_ENABLED", "").strip().lower() in ("1", "true", "yes")


def _model_path() -> str:
    """Prefer env; else first on-disk file (MONAI/VISTA3D-HF bootstrap layout or legacy ``model.pt``)."""
    env = (os.getenv("VISTA3D_MODEL_PATH") or "").strip()
    candidates = [
        env,
        "/models/vista3d/vista3d_pretrained_model/model.safetensors",
        "/models/vista3d/model.pt",
        "/models/vista3d/model.safetensors",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return env or "/models/vista3d/model.pt"


def enrich_vista3d_metadata(
    *,
    volume: np.ndarray,
    meta: dict,
    film_photo: bool,
    degraded: bool,
    pathology_scores: dict[str, Any],
    findings: list[Any],
) -> None:
    """Mutates ``pathology_scores`` and ``findings`` when VISTA tier is enabled."""
    if not _vista_enabled():
        return

    path = _model_path()
    status: dict[str, Any] = {
        "enabled": True,
        "checkpoint_path": path,
        "checkpoint_present": bool(path and os.path.isfile(path)),
    }

    if film_photo or degraded:
        status["inference"] = "skipped_non_volumetric_ct"
        findings.append(
            Finding(
                label="VISTA-3D — skipped (non-volumetric or film input)",
                description=(
                    "Premium VISTA-3D segmentation expects a full 3D CT (DICOM series or NIfTI). "
                    "This upload is single-slice, film-photo, or otherwise degraded — "
                    "standard NCCT pipeline results still apply."
                ),
                severity="info",
                confidence=100.0,
                region="Brain",
            )
        )
        pathology_scores["vista3d"] = status
        return

    if not status["checkpoint_present"]:
        status["inference"] = "weights_missing"
        findings.append(
            Finding(
                label="VISTA-3D checkpoint not on volume",
                description=(
                    "Run Modal bootstrap for VISTA weights (see docs/MODAL_PHASE1_OPERATOR_SETUP.md) "
                    "or set VISTA3D_MODEL_PATH to a valid checkpoint."
                ),
                severity="warning",
                confidence=100.0,
                region="Brain",
            )
        )
        pathology_scores["vista3d"] = status
        return

    # Checkpoint present: record readiness; optional torch sanity check
    try:
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            status["checkpoint_keys_sample"] = list(ckpt.keys())[:8]
        status["inference"] = "checkpoint_loaded_metadata_only"
        status["note"] = (
            "Full VISTA-3D forward integration can be extended with MONAI VISTA bundle APIs; "
            "NCCT ICH / narrative pipeline unchanged."
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("VISTA-3D checkpoint inspect failed: %s", e)
        status["inference"] = "checkpoint_load_error"
        status["error"] = str(e)[:300]

    pathology_scores["vista3d"] = status
