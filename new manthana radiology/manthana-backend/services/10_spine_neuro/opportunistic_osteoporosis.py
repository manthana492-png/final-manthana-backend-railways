"""
Opportunistic osteoporosis screening from L1 vertebra HU.
Uses TotalSegmentator vertebrae_body masks to extract mean HU from L1.

BMD proxy thresholds (validated against DXA):
- >160 HU: normal
- 100-160 HU: osteopenia
- <100 HU: osteoporosis
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.spine_neuro.osteoporosis")


def _load_mask(mask_path: str | None) -> np.ndarray | None:
    """Load NIfTI mask if path provided."""
    if not mask_path:
        return None
    try:
        import nibabel as nib
        nii = nib.load(mask_path)
        return nii.get_fdata().astype(bool)
    except Exception as e:
        logger.debug("Failed to load mask %s: %s", mask_path, e)
        return None


def _get_voxel_volume_mm3(meta: dict) -> float:
    """Get voxel volume in mm3 from metadata."""
    ps = meta.get("pixel_spacing") if isinstance(meta, dict) else None
    st = meta.get("slice_thickness") if isinstance(meta, dict) else None
    
    if isinstance(ps, (list, tuple)) and len(ps) >= 2:
        try:
            pixel_area_mm2 = float(ps[0]) * float(ps[1])
            slice_thickness_mm = float(st) if st is not None else 1.0
            return pixel_area_mm2 * slice_thickness_mm
        except (TypeError, ValueError):
            pass
    return 1.0


def compute_l1_hu_bmd(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
    meta: dict,
) -> dict[str, Any]:
    """
    Compute L1 vertebra HU for opportunistic BMD screening.
    
    Returns:
        l1_hu_mean: Mean HU in L1 mask
        bone_density_category: normal | osteopenia | osteoporosis | unknown
        opportunistic_screen_result: screening result with recommendation
    """
    v = np.asarray(volume_hu, dtype=np.float32)
    
    # Look for L1 vertebra mask
    l1_mask = None
    l1_keys = ["vertebrae_L1", "L1", "vertebrae_L1_body"]
    for key in l1_keys:
        if key in totalseg_masks:
            l1_mask = _load_mask(totalseg_masks[key])
            if l1_mask is not None:
                break
    
    # Also try generic vertebrae mask if no L1-specific
    if l1_mask is None:
        for key in totalseg_masks:
            if "L1" in key.upper() or ("vertebra" in key.lower() and "lumbar" in key.lower()):
                l1_mask = _load_mask(totalseg_masks[key])
                if l1_mask is not None:
                    break
    
    if l1_mask is None:
        return {
            "l1_hu_mean": None,
            "bone_density_category": "unknown",
            "opportunistic_screen_result": None,
            "available": False,
            "reason": "no_l1_mask",
        }
    
    # Ensure shapes match
    if l1_mask.shape != v.shape:
        logger.warning(f"L1 mask shape {l1_mask.shape} does not match volume shape {v.shape}")
        if v.ndim == 3 and l1_mask.ndim == 3:
            # Try to use minimum dimensions
            min_z = min(v.shape[0], l1_mask.shape[0])
            min_y = min(v.shape[1], l1_mask.shape[1])
            min_x = min(v.shape[2], l1_mask.shape[2])
            v = v[:min_z, :min_y, :min_x]
            l1_mask = l1_mask[:min_z, :min_y, :min_x]
        else:
            return {
                "l1_hu_mean": None,
                "bone_density_category": "unknown",
                "opportunistic_screen_result": None,
                "available": False,
                "reason": "shape_mismatch",
            }
    
    # Calculate mean HU in L1
    l1_voxels = v[l1_mask]
    if l1_voxels.size == 0:
        return {
            "l1_hu_mean": None,
            "bone_density_category": "unknown",
            "opportunistic_screen_result": None,
            "available": False,
            "reason": "empty_mask",
        }
    
    l1_hu_mean = float(np.mean(l1_voxels))
    
    # BMD category based on HU thresholds
    # References:
    # - Pickhardt et al. Radiology 2013
    # - Opportunistic CT: >160 = normal, 100-160 = osteopenia, <100 = osteoporosis
    if l1_hu_mean > 160:
        category = "normal"
        recommendation = "Normal BMD by CT. Continue routine screening per guidelines."
    elif l1_hu_mean >= 100:
        category = "osteopenia"
        recommendation = "Low BMD (osteopenia range). Consider DXA for confirmation and clinical correlation."
    else:
        category = "osteoporosis"
        recommendation = "Very low BMD (osteoporosis range). Strongly recommend DXA and clinical evaluation."
    
    # Calculate L1 volume for completeness
    voxel_mm3 = _get_voxel_volume_mm3(meta)
    l1_volume_cm3 = float(np.sum(l1_mask)) * voxel_mm3 / 1000.0
    
    return {
        "l1_hu_mean": round(l1_hu_mean, 1),
        "bone_density_category": category,
        "opportunistic_screen_result": {
            "recommendation": recommendation,
            "l1_volume_cm3": round(l1_volume_cm3, 2),
            "n_voxels": int(np.sum(l1_mask)),
        },
        "available": True,
    }