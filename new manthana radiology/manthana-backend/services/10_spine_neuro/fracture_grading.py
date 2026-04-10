"""
Genant fracture grading from TotalSegmentator vertebra masks.
Uses geometric analysis of mask bounding boxes per level.

Genant grades:
- Grade 0: normal (<20% height loss)
- Grade 1: mild (20-25% height loss)
- Grade 2: moderate (25-40% height loss)
- Grade 3: severe (>40% height loss)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from scipy import ndimage

logger = logging.getLogger("manthana.spine_neuro.fracture")


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


def _extract_vertebra_level(key: str) -> str | None:
    """Extract vertebra level (L1, L2, T12, etc.) from mask key."""
    # Match patterns like "vertebrae_L1", "L1", "vertebrae_T12", "T12", etc.
    patterns = [
        r'(?:vertebrae_)?([LT]\d+)',
        r'([LT]\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, key, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _sort_vertebra_levels(levels: list[str]) -> list[str]:
    """Sort vertebra levels anatomically (T1-T12, L1-L5, S1)."""
    def sort_key(level):
        match = re.match(r'([TLS])(\d+)', level, re.IGNORECASE)
        if not match:
            return (999, 0)
        region, num = match.groups()
        region_order = {'T': 0, 'L': 1, 'S': 2}
        return (region_order.get(region.upper(), 999), int(num))
    return sorted(levels, key=sort_key)


def _measure_vertebral_heights(mask: np.ndarray) -> dict[str, float]:
    """
    Measure vertebral heights from mask using bounding box analysis.
    
    Returns:
        anterior_height_mm: Height of anterior column
        posterior_height_mm: Height of posterior column  
        middle_height_mm: Height of middle column
        compression_ratio: anterior / posterior
    """
    if not np.any(mask):
        return {
            "anterior_height_mm": None,
            "posterior_height_mm": None,
            "middle_height_mm": None,
            "compression_ratio": None,
        }
    
    # Get bounding box coordinates
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return {
            "anterior_height_mm": None,
            "posterior_height_mm": None,
            "middle_height_mm": None,
            "compression_ratio": None,
        }
    
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    # Assuming z=superior-inferior (height), y=anterior-posterior, x=left-right
    # This is a simplification; actual orientation depends on DICOM orientation
    
    # Full height (anterior-posterior extent)
    full_height = float(y_max - y_min + 1)
    
    # Estimate anterior, middle, posterior thirds
    anterior_third = y_min + (y_max - y_min) // 3
    posterior_third = y_max - (y_max - y_min) // 3
    
    # Measure heights at different positions
    # Anterior: front of vertebra
    anterior_mask = mask.copy()
    anterior_mask[:, :int(y_min + (y_max - y_min) * 0.33), :] = False
    anterior_height = _measure_height_at_region(mask, anterior_mask)
    
    # Posterior: back of vertebra (spinal canal side)
    posterior_mask = mask.copy()
    posterior_mask[:, int(y_max - (y_max - y_min) * 0.33):, :] = False
    posterior_height = _measure_height_at_region(mask, posterior_mask)
    
    # Middle: center region
    middle_start = int(y_min + (y_max - y_min) * 0.33)
    middle_end = int(y_max - (y_max - y_min) * 0.33)
    middle_mask = mask.copy()
    middle_mask[:, :middle_start, :] = False
    middle_mask[:, middle_end:, :] = False
    middle_height = _measure_height_at_region(mask, middle_mask)
    
    compression_ratio = anterior_height / max(posterior_height, 1e-6) if anterior_height and posterior_height else None
    
    return {
        "anterior_height_mm": round(anterior_height, 1) if anterior_height else None,
        "posterior_height_mm": round(posterior_height, 1) if posterior_height else None,
        "middle_height_mm": round(middle_height, 1) if middle_height else None,
        "compression_ratio": round(compression_ratio, 2) if compression_ratio else None,
    }


def _measure_height_at_region(full_mask: np.ndarray, region_mask: np.ndarray) -> float | None:
    """Measure height (superior-inferior extent) at a specific region of the mask."""
    intersection = full_mask & region_mask
    if not np.any(intersection):
        return None
    
    # Find z-extent (superior to inferior) for this region
    z_coords = np.where(intersection)[0]
    if len(z_coords) == 0:
        return None
    
    return float(z_coords.max() - z_coords.min() + 1)


def _calculate_genant_grade(
    current_height: float,
    expected_height: float,
    compression_ratio: float | None,
) -> tuple[int, str]:
    """
    Calculate Genant grade based on height loss.
    
    Returns:
        grade: 0-3
        grade_type: wedge | biconcave | crush | indeterminate
    """
    if expected_height <= 0:
        return 0, "indeterminate"
    
    height_loss_pct = (1.0 - current_height / expected_height) * 100.0
    
    # Determine type from compression ratio
    if compression_ratio is not None:
        if compression_ratio < 0.75:
            vtype = "wedge"
        elif compression_ratio > 1.25:
            vtype = "biconcave"
        elif height_loss_pct > 40:
            vtype = "crush"
        else:
            vtype = "indeterminate"
    else:
        vtype = "indeterminate"
    
    # Genant grading
    if height_loss_pct < 20:
        grade = 0
    elif height_loss_pct < 25:
        grade = 1
    elif height_loss_pct <= 40:
        grade = 2
    else:
        grade = 3
    
    return grade, vtype


def grade_vertebral_fractures(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
    meta: dict,
) -> dict[str, Any]:
    """
    Grade vertebral fractures using Genant method.
    
    Args:
        volume_hu: CT volume (for orientation reference)
        totalseg_masks: Dict mapping structure names to mask file paths
        meta: Metadata dict
        
    Returns:
        List of graded vertebrae with heights and fracture grades
    """
    # Find all vertebra masks
    vertebra_masks = {}
    for key, path in totalseg_masks.items():
        level = _extract_vertebra_level(key)
        if level:
            mask = _load_mask(path)
            if mask is not None:
                vertebra_masks[level] = mask
    
    if not vertebra_masks:
        return {
            "vertebrae": [],
            "any_fracture": False,
            "highest_grade": 0,
            "available": False,
            "reason": "no_vertebra_masks",
        }
    
    # Sort anatomically
    sorted_levels = _sort_vertebra_levels(list(vertebra_masks.keys()))
    
    # Get voxel spacing for mm calculations
    ps = meta.get("pixel_spacing") if isinstance(meta, dict) else None
    st = meta.get("slice_thickness") if isinstance(meta, dict) else None
    
    if isinstance(ps, (list, tuple)) and len(ps) >= 2:
        try:
            z_spacing = float(st) if st is not None else float(ps[0])
        except (TypeError, ValueError):
            z_spacing = 1.0
    else:
        z_spacing = 1.0
    
    # Measure all vertebrae
    vertebra_results = []
    heights_mm = []
    
    for level in sorted_levels:
        mask = vertebra_masks[level]
        measurements = _measure_vertebral_heights(mask)
        
        # Use middle height as primary height measurement
        height_mm = measurements.get("middle_height_mm")
        if height_mm is None:
            height_mm = measurements.get("anterior_height_mm")
        if height_mm is None:
            height_mm = measurements.get("posterior_height_mm")
        
        if height_mm:
            height_mm = height_mm * z_spacing
            heights_mm.append(height_mm)
        
        vertebra_results.append({
            "level": level,
            "heights_mm": {
                "anterior": measurements.get("anterior_height_mm", 0) * z_spacing if measurements.get("anterior_height_mm") else None,
                "posterior": measurements.get("posterior_height_mm", 0) * z_spacing if measurements.get("posterior_height_mm") else None,
                "middle": measurements.get("middle_height_mm", 0) * z_spacing if measurements.get("middle_height_mm") else None,
            },
            "compression_ratio": measurements.get("compression_ratio"),
            "height_mm": round(height_mm, 1) if height_mm else None,
            "genant_grade": None,  # Will be calculated after we have reference heights
            "genant_type": None,
        })
    
    # Calculate Genant grades using adjacent vertebrae as reference
    # For each vertebra, use average of adjacent as expected normal height
    for i, v in enumerate(vertebra_results):
        if v["height_mm"] is None:
            continue
        
        # Get adjacent heights
        adjacent_heights = []
        if i > 0 and vertebra_results[i - 1]["height_mm"]:
            adjacent_heights.append(vertebra_results[i - 1]["height_mm"])
        if i < len(vertebra_results) - 1 and vertebra_results[i + 1]["height_mm"]:
            adjacent_heights.append(vertebra_results[i + 1]["height_mm"])
        
        if adjacent_heights:
            expected_height = sum(adjacent_heights) / len(adjacent_heights)
            grade, vtype = _calculate_genant_grade(
                v["height_mm"],
                expected_height,
                v["compression_ratio"],
            )
            v["genant_grade"] = grade
            v["genant_type"] = vtype
            v["height_loss_pct"] = round((1.0 - v["height_mm"] / expected_height) * 100, 1)
            v["expected_height_mm"] = round(expected_height, 1)
    
    # Summary statistics
    any_fracture = any(v.get("genant_grade", 0) and v["genant_grade"] >= 1 for v in vertebra_results)
    highest_grade = max((v.get("genant_grade", 0) or 0) for v in vertebra_results) if vertebra_results else 0
    
    return {
        "vertebrae": vertebra_results,
        "any_fracture": any_fracture,
        "highest_grade": highest_grade,
        "available": True,
        "n_vertebrae_measured": len(vertebra_results),
    }