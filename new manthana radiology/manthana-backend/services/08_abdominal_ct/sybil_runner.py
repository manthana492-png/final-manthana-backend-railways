"""
Sybil lung cancer risk runner for chest CT DICOM series.
6-year risk prediction using MIT/SimbioSys Sybil model.
Graceful fallback if sybil not installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("manthana.sybil")


def _discover_dicom_paths(series_dir: str) -> list[str]:
    """Discover DICOM files in series directory, sorted."""
    p = Path(series_dir)
    paths = []
    for ext in (".dcm", ".dic", ".dicom"):
        paths.extend(p.glob(f"*{ext}"))
        paths.extend(p.glob(f"*{ext.upper()}"))
    # Also check files without extension
    for f in p.iterdir():
        if f.is_file() and f.suffix == "":
            paths.append(f)
    return sorted([str(x) for x in paths if x.is_file()])


def run_sybil(dicom_paths: list[str]) -> dict[str, Any]:
    """
    Run Sybil 6-year lung cancer risk prediction.
    
    Args:
        dicom_paths: List of DICOM file paths from chest CT series
        
    Returns:
        dict with risk_1yr through risk_6yr, risk_category, recommend_followup_ldct
        If sybil not installed, returns {"available": False, "reason": "sybil_not_installed"}
    """
    try:
        from sybil import Serie, Sybil
    except ImportError:
        logger.info("Sybil package not installed; lung cancer risk prediction unavailable")
        return {"available": False, "reason": "sybil_not_installed"}
    
    if not dicom_paths:
        return {"available": False, "reason": "no_dicom_paths"}
    
    try:
        # Use ensemble model by default
        model_name = os.getenv("SYBIL_MODEL", "sybil_ensemble")
        model = Sybil(model_name)
        
        # Create Serie from DICOM paths
        serie = Serie(dicom_paths)
        
        # Predict returns list of predictions per serie
        predictions = model.predict([serie])
        
        if not predictions or len(predictions) == 0:
            return {"available": False, "reason": "prediction_failed"}
        
        pred = predictions[0]
        
        # Extract 6-year risks
        risks = {}
        for year in range(1, 7):
            key = f"risk_{year}yr"
            if hasattr(pred, key):
                risks[key] = float(getattr(pred, key))
            elif hasattr(pred, "risks") and len(pred.risks) >= year:
                risks[key] = float(pred.risks[year - 1])
            else:
                risks[key] = None
        
        # Determine risk category
        risk_1yr = risks.get("risk_1yr", 0.0) or 0.0
        risk_6yr = risks.get("risk_6yr", 0.0) or 0.0
        
        if risk_6yr < 0.015:
            risk_category = "low"
        elif risk_6yr < 0.05:
            risk_category = "moderate"
        elif risk_6yr < 0.10:
            risk_category = "high"
        else:
            risk_category = "very_high"
        
        # Follow-up recommendation
        recommend_ldct = risk_category in ("high", "very_high") or risk_1yr > 0.02
        
        return {
            "available": True,
            "model": model_name,
            **risks,
            "risk_category": risk_category,
            "recommend_followup_ldct": recommend_ldct,
            "n_slices": len(dicom_paths),
        }
        
    except Exception as e:
        logger.warning("Sybil prediction failed: %s", e)
        return {"available": False, "reason": f"prediction_error: {type(e).__name__}"}


def is_sybil_available() -> bool:
    """Check if sybil package is installed and functional."""
    try:
        import sybil
        return True
    except ImportError:
        return False