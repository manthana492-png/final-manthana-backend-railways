"""
Prima end-to-end MRI study pipeline (MLNeurosurg/Prima).

Production default: unavailable until PRIMA_CONFIG_YAML and mounted weights exist.
Spike A documents: python end-to-end_inference_pipeline/pipeline.py --config <yaml>
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("manthana.prima_pipeline")

# Curated summary keys (when Prima returns 52-class logits)
PRIMA_SUMMARY_LABELS = frozenset(
    {
        "normal",
        "mass_lesion",
        "hemorrhage",
        "infarct",
        "white_matter_disease",
        "hydrocephalus",
        "atrophy",
        "enhancement",
        "midline_shift",
        "edema",
        "vascular_abnormality",
        "infection_inflammation",
    }
)


def run_prima_study(
    study_dir_or_nifti: str,
    job_id: str,
) -> dict[str, Any]:
    """
    Run Prima inference when configured. Otherwise returns available=False with empty scores.
    """
    cfg = os.getenv("PRIMA_CONFIG_YAML", "").strip()
    repo = os.getenv("PRIMA_REPO_DIR", "/opt/Prima").strip()
    if not cfg or not os.path.isfile(cfg):
        logger.debug("Prima not configured (PRIMA_CONFIG_YAML missing) — skipping")
        return {"available": False, "scores": {}, "reason": "prima_not_configured"}
    if not os.path.isdir(repo):
        return {"available": False, "scores": {}, "reason": "prima_repo_missing"}

    # Future: subprocess to end-to-end_inference_pipeline/pipeline.py --config cfg
    logger.info("Prima weights path configured but subprocess not wired — skipping inference")
    return {"available": False, "scores": {}, "reason": "prima_pipeline_not_implemented"}
