"""
Cardiac heuristics from TotalSegmentator heartchamber masks.
- Pericardial effusion: pericardium minus heart mask volume delta
- Cardiomegaly index: cardiac:thoracic volume ratio
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.cardiac_ct.heuristics")


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


def _mask_volume_cm3(mask: np.ndarray, voxel_mm3: float) -> float:
    """Calculate volume in cm3 from binary mask."""
    return float(np.sum(mask)) * voxel_mm3 / 1000.0


def compute_pericardial_effusion(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
    meta: dict,
) -> dict[str, Any]:
    """
    Detect pericardial effusion by comparing pericardium volume to heart chamber volume.
    
    Returns effusion flag and volume delta.
    """
    # Get pericardium mask if available
    pericardium_mask = None
    if "pericardium" in totalseg_masks:
        pericardium_mask = _load_mask(totalseg_masks["pericardium"])
    
    # Build heart chambers mask
    heart_keys = [
        "heart_left_ventricle",
        "heart_right_ventricle",
        "heart_left_atrium",
        "heart_right_atrium",
        "heart_myocardium",
    ]
    heart_mask = None
    for key in heart_keys:
        if key in totalseg_masks:
            m = _load_mask(totalseg_masks[key])
            if m is not None:
                if heart_mask is None:
                    heart_mask = m
                else:
                    heart_mask = heart_mask | m
    
    if pericardium_mask is None or heart_mask is None:
        return {
            "pericardial_effusion_suspected": None,
            "pericardial_effusion_volume_ml": None,
            "pericardial_fat_volume_ml": None,
            "available": False,
        }
    
    # Ensure same shape
    if pericardium_mask.shape != heart_mask.shape:
        return {
            "pericardial_effusion_suspected": None,
            "pericardial_effusion_volume_ml": None,
            "pericardial_fat_volume_ml": None,
            "available": False,
            "reason": "shape_mismatch",
        }
    
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
        pixel_area_mm2 = 1.0
        slice_thickness_mm = 1.0
    
    voxel_mm3 = pixel_area_mm2 * slice_thickness_mm
    
    # Calculate volumes
    pericardium_volume_ml = _mask_volume_cm3(pericardium_mask, voxel_mm3) * 10  # Convert cm3 to ml (1 cm3 = 1 ml, but using factor for correction)
    # Actually cm3 = ml, so no conversion needed
    pericardium_volume_ml = _mask_volume_cm3(pericardium_mask, voxel_mm3)
    heart_volume_ml = _mask_volume_cm3(heart_mask, voxel_mm3)
    
    # Pericardial space = pericardium - heart chambers
    pericardial_space_mask = pericardium_mask & ~heart_mask
    pericardial_space_ml = _mask_volume_cm3(pericardial_space_mask, voxel_mm3)
    
    # Fat attenuation in pericardial space (negative HU)
    v = np.asarray(volume_hu, dtype=np.float32)
    if v.shape == pericardial_space_mask.shape:
        pericardial_fat = pericardial_space_mask & (v < -50)  # Fat HU < -50
        pericardial_fat_ml = _mask_volume_cm3(pericardial_fat, voxel_mm3)
        
        # Effusion = fluid attenuation in pericardial space (0-20 HU)
        pericardial_fluid = pericardial_space_mask & (v > -20) & (v < 20)
        pericardial_fluid_ml = _mask_volume_cm3(pericardial_fluid, voxel_mm3)
    else:
        pericardial_fat_ml = None
        pericardial_fluid_ml = None
    
    # Suspect effusion if pericardial space volume > threshold
    # Normal pericardial space has minimal fluid (<15 ml typically)
    effusion_threshold_ml = 15.0
    effusion_suspected = pericardial_fluid_ml is not None and pericardial_fluid_ml > effusion_threshold_ml
    
    return {
        "pericardial_effusion_suspected": effusion_suspected,
        "pericardial_effusion_volume_ml": round(pericardial_fluid_ml, 2) if pericardial_fluid_ml is not None else None,
        "pericardial_fat_volume_ml": round(pericardial_fat_ml, 2) if pericardial_fat_ml is not None else None,
        "pericardial_total_space_ml": round(pericardial_space_ml, 2),
        "heart_volume_ml": round(heart_volume_ml, 2),
        "available": True,
    }


def compute_cardiomegaly_index(
    totalseg_masks: dict[str, str],
    meta: dict,
) -> dict[str, Any]:
    """
    Calculate cardiomegaly index (cardiac:thoracic volume ratio).
    
    Cardiomegaly index > 0.5 suggests cardiomegaly (simplified CT adaptation of CTR).
    """
    # Get heart chambers mask
    heart_keys = [
        "heart_left_ventricle",
        "heart_right_ventricle",
        "heart_left_atrium",
        "heart_right_atrium",
        "heart_myocardium",
    ]
    heart_mask = None
    for key in heart_keys:
        if key in totalseg_masks:
            m = _load_mask(totalseg_masks[key])
            if m is not None:
                if heart_mask is None:
                    heart_mask = m
                else:
                    heart_mask = heart_mask | m
    
    # Get thorax/lung mask
    thorax_mask = None
    thorax_keys = ["lung_left", "lung_right", "thorax", "ribcage"]
    for key in thorax_keys:
        if key in totalseg_masks:
            m = _load_mask(totalseg_masks[key])
            if m is not None:
                if thorax_mask is None:
                    thorax_mask = m
                else:
                    thorax_mask = thorax_mask | m
    
    if heart_mask is None or thorax_mask is None:
        return {
            "cardiomegaly_index": None,
            "cardiomegaly_suspected": None,
            "heart_volume_ml": None,
            "thoracic_volume_ml": None,
            "available": False,
        }
    
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
        pixel_area_mm2 = 1.0
        slice_thickness_mm = 1.0
    
    voxel_mm3 = pixel_area_mm2 * slice_thickness_mm
    
    # Calculate volumes
    heart_volume_ml = _mask_volume_cm3(heart_mask, voxel_mm3)
    thoracic_volume_ml = _mask_volume_cm3(thorax_mask, voxel_mm3)
    
    # Cardiomegaly index = heart volume / thoracic volume
    if thoracic_volume_ml > 0:
        cardiomegaly_index = heart_volume_ml / thoracic_volume_ml
    else:
        cardiomegaly_index = None
    
    # Threshold for cardiomegaly (simplified, normally use cardiothoracic ratio on PA chest X-ray)
    # For CT, this is an approximation
    cardiomegaly_threshold = 0.35  # Adjusted for CT volumes
    cardiomegaly_suspected = cardiomegaly_index is not None and cardiomegaly_index > cardiomegaly_threshold
    
    return {
        "cardiomegaly_index": round(cardiomegaly_index, 3) if cardiomegaly_index is not None else None,
        "cardiomegaly_suspected": cardiomegaly_suspected,
        "heart_volume_ml": round(heart_volume_ml, 2),
        "thoracic_volume_ml": round(thoracic_volume_ml, 2),
        "available": True,
    }


def run_cardiac_heuristics(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
    meta: dict,
) -> dict[str, Any]:
    """
    Run all cardiac heuristics.
    
    Returns combined pericardial and cardiomegaly metrics.
    """
    pericardial = compute_pericardial_effusion(volume_hu, totalseg_masks, meta)
    cardiomegaly = compute_cardiomegaly_index(totalseg_masks, meta)
    
    return {
        "pericardial": pericardial,
        "cardiomegaly": cardiomegaly,
    }