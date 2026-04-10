"""
Coronary Artery Calcium (CAC) scoring from non-gated CT.
Agatston score using HU thresholds on cardiac region masks from TotalSegmentator.

HU thresholds:
- >130 HU = calcium detection in coronary ROI
- No external model required -- pure HU + mask arithmetic
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.cardiac_ct.cac")


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


def _get_coronary_roi_mask(totalseg_masks: dict[str, str]) -> np.ndarray | None:
    """
    Build coronary ROI mask from TotalSegmentator heart chamber masks.
    Combines left/right ventricle, left/right atrium as coronary ROI proxy.
    """
    heart_keys = [
        "heart_left_ventricle",
        "heart_right_ventricle",
        "heart_left_atrium",
        "heart_right_atrium",
        "heart_myocardium",
        "aorta",
        "pulmonary_artery",
    ]
    
    combined = None
    for key in heart_keys:
        if key in totalseg_masks:
            mask = _load_mask(totalseg_masks[key])
            if mask is not None:
                if combined is None:
                    combined = mask
                else:
                    combined = combined | mask
    
    return combined


def _compute_agatston_for_slice(slice_hu: np.ndarray, slice_mask: np.ndarray, pixel_area_mm2: float) -> float:
    """
    Compute Agatston score for a single slice.
    
    Score = area (mm2) * max(HU) coefficient
    Coefficients: 130-199 HU = 1, 200-299 = 2, 300-399 = 3, >=400 = 4
    """
    if not np.any(slice_mask):
        return 0.0
    
    # Get calcium regions (>130 HU)
    calcium = (slice_hu > 130) & slice_mask
    if not np.any(calcium):
        return 0.0
    
    # Label connected components
    from scipy import ndimage
    labeled, n_features = ndimage.label(calcium.astype(np.uint8))
    
    score = 0.0
    for i in range(1, n_features + 1):
        component = labeled == i
        max_hu = float(np.max(slice_hu[component]))
        area_mm2 = np.sum(component) * pixel_area_mm2
        
        # Determine coefficient
        if max_hu >= 400:
            coef = 4
        elif max_hu >= 300:
            coef = 3
        elif max_hu >= 200:
            coef = 2
        else:
            coef = 1
        
        score += area_mm2 * coef
    
    return score


def run_calcium_scoring(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
    meta: dict,
) -> dict[str, Any]:
    """
    Calculate Agatston CAC score from non-gated CT.
    
    Args:
        volume_hu: CT volume in Hounsfield Units
        totalseg_masks: Dict mapping structure names to mask file paths from TotalSeg
        meta: DICOM metadata dict with pixel_spacing and slice_thickness
        
    Returns:
        Dict with cac_agatston_score, cac_risk_category, cac_volume_mm3
    """
    v = np.asarray(volume_hu, dtype=np.float32)
    
    # Get voxel spacing
    ps = meta.get("pixel_spacing") if isinstance(meta, dict) else None
    st = meta.get("slice_thickness") if isinstance(meta, dict) else None
    
    if isinstance(ps, (list, tuple)) and len(ps) >= 2:
        try:
            pixel_area_mm2 = float(ps[0]) * float(ps[1])
            slice_thickness_mm = float(st) if st is not None else 1.0
        except (TypeError, ValueError):
            pixel_area_mm2 = 1.0
            slice_thickness_mm = 1.0
    else:
        # Default 1mm isotropic
        pixel_area_mm2 = 1.0
        slice_thickness_mm = 1.0
    
    voxel_mm3 = pixel_area_mm2 * slice_thickness_mm
    
    # Get coronary ROI mask
    coronary_mask = _get_coronary_roi_mask(totalseg_masks)
    if coronary_mask is None:
        return {
            "cac_agatston_score": None,
            "cac_risk_category": "unknown",
            "cac_volume_mm3": None,
            "cac_available": False,
            "cac_reason": "no_heart_masks",
        }
    
    # Ensure mask matches volume shape
    if coronary_mask.shape != v.shape:
        logger.warning(f"Mask shape {coronary_mask.shape} does not match volume shape {v.shape}")
        # Try to handle 2D vs 3D mismatch
        if v.ndim == 3 and coronary_mask.ndim == 3:
            # Use minimum dimensions
            min_z = min(v.shape[0], coronary_mask.shape[0])
            min_y = min(v.shape[1], coronary_mask.shape[1])
            min_x = min(v.shape[2], coronary_mask.shape[2])
            v = v[:min_z, :min_y, :min_x]
            coronary_mask = coronary_mask[:min_z, :min_y, :min_x]
        else:
            return {
                "cac_agatston_score": None,
                "cac_risk_category": "unknown",
                "cac_volume_mm3": None,
                "cac_available": False,
                "cac_reason": "shape_mismatch",
            }
    
    # Compute Agatston score slice by slice
    total_score = 0.0
    calcium_volume_voxels = 0
    
    for z in range(v.shape[0]):
        slice_hu = v[z]
        slice_mask = coronary_mask[z]
        
        # Detect calcium (>130 HU)
        calcium = (slice_hu > 130) & slice_mask
        if np.any(calcium):
            slice_score = _compute_agatston_for_slice(slice_hu, slice_mask, pixel_area_mm2)
            total_score += slice_score
            calcium_volume_voxels += np.sum(calcium)
    
    # Risk categories: 0=none, 1-100=mild, 101-400=moderate, >400=severe
    if total_score == 0:
        risk_category = "none"
    elif total_score <= 100:
        risk_category = "mild"
    elif total_score <= 400:
        risk_category = "moderate"
    else:
        risk_category = "severe"
    
    calcium_volume_mm3 = float(calcium_volume_voxels) * voxel_mm3
    
    return {
        "cac_agatston_score": round(total_score, 1),
        "cac_risk_category": risk_category,
        "cac_volume_mm3": round(calcium_volume_mm3, 2),
        "cac_available": True,
    }