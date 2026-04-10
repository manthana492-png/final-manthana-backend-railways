"""
White matter hyperintensity (WMH) quantification.
Fazekas score derivation from volume or dedicated segmentation model.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.brain_mri.wmh")

_wmh_model: Any = None


def _load_wmh_model() -> Any | None:
    """Load WMH segmentation model if available."""
    global _wmh_model
    path = (os.environ.get("WMH_MODEL_PATH") or "").strip()
    if not path or not os.path.isfile(path):
        return None
    if _wmh_model is not None:
        return _wmh_model
    
    try:
        import torch
        _wmh_model = torch.jit.load(path, map_location="cpu")
        _wmh_model.eval()
        logger.info("Loaded WMH segmentation TorchScript from %s", path)
        return _wmh_model
    except Exception as e:
        logger.debug("WMH TorchScript load failed: %s", e)
    
    return None


def _estimate_fazekas(wmh_volume_ml: float) -> tuple[int, str]:
    """
    Estimate Fazekas score from WMH volume.
    
    Simplified thresholds:
    - 0: < 10 ml (none)
    - 1: 10-25 ml (mild)
    - 2: 25-50 ml (moderate)
    - 3: > 50 ml (severe)
    
    Distribution:
    - PV (periventricular): >50% adjacent to ventricles
    - Deep: <50% adjacent to ventricles
    """
    if wmh_volume_ml < 10:
        score = 0
    elif wmh_volume_ml < 25:
        score = 1
    elif wmh_volume_ml < 50:
        score = 2
    else:
        score = 3
    
    # Simple heuristic: if volume is large, more likely deep; if small, more likely periventricular
    if wmh_volume_ml < 5:
        distribution = "periventricular"
    elif wmh_volume_ml > 30:
        distribution = "mixed"
    else:
        distribution = "periventricular"  # Default assumption
    
    return score, distribution


def run_wmh_analysis(
    volume: np.ndarray,
    ventricle_mask: np.ndarray | None = None,
    synthseg_volumes: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Run WMH analysis.
    
    Args:
        volume: MRI volume (FLAIR preferred, T2 acceptable)
        ventricle_mask: Binary mask of ventricles for localization
        synthseg_volumes: SynthSeg volumes dict (may include wm_hyperintensity_cm3)
        
    Returns:
        {
            wmh_volume_ml,
            fazekas_score (0-3),
            distribution (periventricular|deep|mixed),
            available
        }
    """
    # Try dedicated model first
    model = _load_wmh_model()
    
    if model is not None:
        try:
            # Preprocess
            v = np.asarray(volume, dtype=np.float32)
            
            # Simple normalization
            p1, p99 = np.percentile(v, [1, 99])
            v = np.clip(v, p1, p99)
            v = (v - v.mean()) / max(v.std(), 1e-6)
            
            # Resize for model
            target_shape = (128, 128, 128)
            if v.shape != target_shape:
                import scipy.ndimage
                zoom_factors = [t / o for t, o in zip(target_shape, v.shape)]
                v = scipy.ndimage.zoom(v, zoom_factors, order=1)
            
            v = v[np.newaxis, np.newaxis, ...]
            
            # Run inference
            import torch
            with torch.no_grad():
                input_tensor = torch.from_numpy(v)
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                pred = (output.numpy()[0, 0] > 0.5).astype(np.uint8)
            
            # Resize back
            if pred.shape != volume.shape:
                import scipy.ndimage
                zoom_factors = [o / t for o, t in zip(volume.shape, pred.shape)]
                pred = scipy.ndimage.zoom(pred, zoom_factors, order=0)
            
            # Calculate volume
            # Assume 1mm isotropic voxels for now
            voxel_mm3 = 1.0
            wmh_volume_ml = float(np.sum(pred)) * voxel_mm3 / 1000.0
            
            fazekas, distribution = _estimate_fazekas(wmh_volume_ml)
            
            return {
                "available": True,
                "model": "dedicated",
                "wmh_volume_ml": round(wmh_volume_ml, 2),
                "fazekas_score": fazekas,
                "distribution": distribution,
            }
            
        except Exception as e:
            logger.warning("WMH model inference failed: %s", e)
            # Fall through to volume-based estimation
    
    # Fallback: use SynthSeg WMH volume if available
    if synthseg_volumes and "wm_hyperintensity_cm3" in synthseg_volumes:
        wmh_cm3 = float(synthseg_volumes["wm_hyperintensity_cm3"])
        wmh_ml = wmh_cm3  # 1 cm3 = 1 ml
        fazekas, distribution = _estimate_fazekas(wmh_ml)
        
        return {
            "available": True,
            "model": "synthseg_proxy",
            "wmh_volume_ml": round(wmh_ml, 2),
            "fazekas_score": fazekas,
            "distribution": distribution,
        }
    
    # No data available
    return {
        "available": False,
        "reason": "no_model_or_synthseg_data",
        "wmh_volume_ml": None,
        "fazekas_score": None,
        "distribution": None,
    }