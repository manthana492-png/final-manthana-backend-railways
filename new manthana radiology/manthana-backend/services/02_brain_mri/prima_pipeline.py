"""
Prima pipeline wrapper (stub for 02_brain_mri service).
Actual implementation is in shared/prima_pipeline.py.
This module provides the run_prima_study interface for local imports.

To enable full functionality:
- Set PRIMA_REPO_DIR (default /opt/Prima)
- Set PRIMA_CONFIG_YAML with valid config path
- Ensure model weights are mounted at expected paths
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("manthana.brain_mri.prima")


def run_prima_study(series_dir_or_filepath: str, job_id: str) -> dict[str, Any]:
    """
    Run Prima pipeline on brain MRI study.
    
    Args:
        series_dir_or_filepath: Path to DICOM series or NIfTI file
        job_id: Job identifier
        
    Returns:
        Dict with available flag and scores, or unavailable with reason
    """
    # Check if Prima is configured
    prima_cfg = os.getenv("PRIMA_CONFIG_YAML", "").strip()
    prima_repo = os.getenv("PRIMA_REPO_DIR", "/opt/Prima")
    
    if not prima_cfg or not os.path.isfile(prima_cfg):
        return {
            "available": False,
            "reason": "prima_not_configured",
            "message": "PRIMA_CONFIG_YAML not set or file not found",
            "scores": {},
        }
    
    # Check if weights exist
    weights_path = os.path.join(prima_repo, "primafullmodel107.pt")
    if not os.path.isfile(weights_path):
        return {
            "available": False,
            "reason": "weights_not_found",
            "message": f"Prima weights not found at {weights_path}",
            "scores": {},
        }
    
    # Try to run actual Prima pipeline from shared module
    try:
        import sys
        shared_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
        if shared_path not in sys.path:
            sys.path.insert(0, shared_path)
        
        from shared.prima_pipeline import run_prima_study as real_prima
        return real_prima(series_dir_or_filepath, job_id)
    except Exception as e:
        logger.warning("Prima pipeline execution failed: %s", e)
        return {
            "available": False,
            "reason": "execution_failed",
            "message": str(e),
            "scores": {},
        }


def is_prima_available() -> bool:
    """Check if Prima pipeline is available (configured and weights present)."""
    prima_cfg = os.getenv("PRIMA_CONFIG_YAML", "").strip()
    prima_repo = os.getenv("PRIMA_REPO_DIR", "/opt/Prima")
    
    if not prima_cfg or not os.path.isfile(prima_cfg):
        return False
    
    weights_path = os.path.join(prima_repo, "primafullmodel107.pt")
    return os.path.isfile(weights_path)