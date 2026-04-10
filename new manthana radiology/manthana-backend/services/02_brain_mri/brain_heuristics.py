"""
Brain MRI heuristics for India population.
- Hydrocephalus detection (Evans index proxy)
- Atrophy pattern (hippocampal ratio for Alzheimer's screening)
- TB meningitis pattern (basal cistern changes)
- NCC ring lesion proxy on T2/FLAIR
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import ndimage

logger = logging.getLogger("manthana.brain_mri.heuristics")


def detect_hydrocephalus(
    ventricle_volume_ml: float | None,
    brain_volume_ml: float | None,
    ventricle_mask: np.ndarray | None = None,
    brain_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Detect hydrocephalus using Evans index proxy (ventricle:brain ratio).
    
    Evans index > 0.3 suggests hydrocephalus (simplified for volumetric ratio).
    """
    if ventricle_volume_ml is None or brain_volume_ml is None:
        if ventricle_mask is None or brain_mask is None:
            return {
                "hydrocephalus_suspected": None,
                "evans_index_proxy": None,
                "ventricle_volume_ml": ventricle_volume_ml,
                "brain_volume_ml": brain_volume_ml,
            }
        
        # Calculate from masks
        vox_vent = float(np.sum(ventricle_mask))
        vox_brain = float(np.sum(brain_mask))
        
        if vox_brain == 0:
            return {
                "hydrocephalus_suspected": None,
                "evans_index_proxy": None,
                "ventricle_volume_ml": None,
                "brain_volume_ml": None,
            }
        
        ratio = vox_vent / vox_brain
    else:
        if brain_volume_ml == 0:
            return {
                "hydrocephalus_suspected": None,
                "evans_index_proxy": None,
                "ventricle_volume_ml": ventricle_volume_ml,
                "brain_volume_ml": brain_volume_ml,
            }
        ratio = ventricle_volume_ml / brain_volume_ml
    
    # Evans index > 0.3 suggests hydrocephalus
    # Using volumetric ratio as proxy (approximation)
    hydrocephalus_threshold = 0.15  # Volumetric ratio is different from linear Evans index
    
    return {
        "hydrocephalus_suspected": ratio > hydrocephalus_threshold,
        "evans_index_proxy": round(ratio, 3),
        "ventricle_volume_ml": ventricle_volume_ml,
        "brain_volume_ml": brain_volume_ml,
    }


def detect_atrophy_pattern(
    synthseg_volumes: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Detect atrophy pattern for Alzheimer's screening using hippocampal ratio.
    
    Uses hippocampal volume : total brain ratio as proxy.
    """
    if synthseg_volumes is None:
        return {
            "atrophy_detected": None,
            "hippocampal_ratio": None,
            "pattern": None,
        }
    
    # Get hippocampal volumes if available
    hipp_left = synthseg_volumes.get("hippocampus_left", 0)
    hipp_right = synthseg_volumes.get("hippocampus_right", 0)
    total_brain = synthseg_volumes.get("brain", 0) or synthseg_volumes.get("total_brain", 0)
    
    if total_brain == 0 or (hipp_left == 0 and hipp_right == 0):
        return {
            "atrophy_detected": None,
            "hippocampal_ratio": None,
            "pattern": None,
        }
    
    hipp_total = hipp_left + hipp_right
    ratio = hipp_total / total_brain
    
    # Thresholds (approximate)
    # Normal hippocampal ratio ~ 0.005-0.007
    # Reduced suggests atrophy
    atrophy_threshold = 0.004
    
    pattern = None
    if ratio < atrophy_threshold:
        atrophy_detected = True
        # Try to determine pattern
        if synthseg_volumes.get("ventricle_volume", 0) / max(total_brain, 1) > 0.15:
            pattern = "hippocampal_and_ventricular"
        else:
            pattern = "hippocampal_predominant"
    else:
        atrophy_detected = False
    
    return {
        "atrophy_detected": atrophy_detected,
        "hippocampal_ratio": round(ratio, 5),
        "pattern": pattern,
    }


def detect_tb_meningitis_pattern(
    volume: np.ndarray,
    ventricle_volume_ml: float | None = None,
    basal_cistern_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Detect TB meningitis pattern based on:
    - Hydrocephalus pattern
    - Basal cistern signal changes (hyperintensity on T2/FLAIR, enhancement on T1 post-contrast)
    
    This is a heuristic proxy - not definitive diagnosis.
    """
    v = np.asarray(volume, dtype=np.float32)
    
    # Check for hydrocephalus pattern
    hydrocephalus = ventricle_volume_ml is not None and ventricle_volume_ml > 50
    
    # Check basal cistern hyperintensity (T2/FLAIR proxy)
    cistern_signal = None
    if basal_cistern_mask is not None and np.any(basal_cistern_mask):
        cistern_voxels = v[basal_cistern_mask]
        cistern_signal = float(np.mean(cistern_voxels))
        
        # T2/FLAIR hyperintensity in cisterns suggests exudate
        # This is sequence-dependent; assume T2/FLAIR if intensity is high
        cistern_hyperintense = cistern_signal > np.percentile(v, 75)
    else:
        cistern_hyperintense = None
    
    # TB meningitis suspicion requires both hydrocephalus and cistern changes
    # or strong hydrocephalus with basal predominance
    tb_suspect = False
    score = 0.0
    
    if hydrocephalus:
        score += 0.4
    if cistern_hyperintense:
        score += 0.3
    if cistern_hyperintense and hydrocephalus:
        score += 0.2
    
    tb_suspect = score >= 0.5
    
    return {
        "tb_meningitis_suspected": tb_suspect,
        "score": round(score, 2),
        "hydrocephalus_present": hydrocephalus,
        "basal_cistern_hyperintensity": cistern_hyperintense,
        "cistern_mean_signal": round(cistern_signal, 1) if cistern_signal else None,
    }


def detect_ncc_ring_lesion(
    volume: np.ndarray,
    t1_volume: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Detect NCC (neurocysticercosis) ring lesion pattern on T2/FLAIR.
    
    Characteristic features:
    - Cystic lesions with scolex (T1 hyperintense dot)
    - Ring enhancement on post-contrast T1
    - Variable T2 signal depending on stage
    
    This is a simplified heuristic for cystic ring-like lesions.
    """
    v = np.asarray(volume, dtype=np.float32)
    
    # Detect cystic regions (CSF-like signal on T2)
    # NCC cysts are typically hypointense to brain on T2 with hyperintense wall
    cystic_candidates = (v > 10) & (v < 60)  # CSF-like signal range
    
    # Look for ring-like structures using morphological operations
    from scipy import ndimage
    
    # Find connected components
    labeled, n_features = ndimage.label(cystic_candidates.astype(np.uint8))
    
    ring_lesions = []
    
    for i in range(1, n_features + 1):
        component = labeled == i
        size = np.sum(component)
        
        # NCC cysts are typically 5-20mm
        # Assuming 1mm voxels, size range 125-8000 voxels
        if size < 50 or size > 10000:
            continue
        
        # Check for ring structure: hollow center with bright rim
        eroded = ndimage.binary_erosion(component, iterations=2)
        ring = component & ~eroded
        
        if np.sum(ring) > np.sum(eroded) * 0.3:  # Significant ring component
            # Check rim intensity
            rim_voxels = v[ring]
            rim_mean = np.mean(rim_voxels)
            center_mean = np.mean(v[eroded]) if np.any(eroded) else 0
            
            # Ring should be brighter than center (inflammatory rim)
            if rim_mean > center_mean + 10:
                # Get centroid location
                coords = np.where(component)
                centroid = tuple(np.mean(c) for c in coords)
                
                ring_lesions.append({
                    "centroid": centroid,
                    "size_voxels": int(size),
                    "rim_intensity": round(float(rim_mean), 1),
                    "center_intensity": round(float(center_mean), 1),
                })
    
    # Scoring
    n_lesions = len(ring_lesions)
    ncc_suspect = n_lesions >= 2 or (n_lesions == 1 and ring_lesions[0]["size_voxels"] > 200)
    
    return {
        "ncc_suspected": ncc_suspect,
        "ring_lesion_count": n_lesions,
        "lesions": ring_lesions[:10],  # Limit to top 10
        "stage_hint": "viable" if n_lesions > 0 and all(l["center_intensity"] < 30 for l in ring_lesions) else "indeterminate",
    }


def run_brain_heuristics(
    volume: np.ndarray,
    synthseg_volumes: dict[str, float] | None = None,
    ventricle_volume_ml: float | None = None,
    brain_volume_ml: float | None = None,
    t1_volume: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Run all brain MRI heuristics.
    
    Returns combined results for hydrocephalus, atrophy, TB meningitis, NCC.
    """
    hydrocephalus = detect_hydrocephalus(ventricle_volume_ml, brain_volume_ml)
    atrophy = detect_atrophy_pattern(synthseg_volumes)
    tb_meningitis = detect_tb_meningitis_pattern(volume, ventricle_volume_ml)
    ncc = detect_ncc_ring_lesion(volume, t1_volume)
    
    return {
        "hydrocephalus": hydrocephalus,
        "atrophy": atrophy,
        "tb_meningitis": tb_meningitis,
        "ncc": ncc,
    }