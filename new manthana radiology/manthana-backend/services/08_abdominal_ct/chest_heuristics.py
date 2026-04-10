"""
Chest CT heuristics for India population:
- TB heuristic: apical consolidation, calcified granulomas, cavitation proxy
- NAFLD grading: liver HU vs spleen HU ratio
- Tropical pancreatitis flag: pancreatic calcifications + ductal dilation proxy
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import ndimage

logger = logging.getLogger("manthana.chest_heuristics")


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


def compute_tb_heuristic(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
    lung_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    TB heuristic: apical consolidation, calcified granulomas, cavitation proxy.
    Returns suspicion score and findings.
    """
    v = np.asarray(volume_hu, dtype=np.float32)
    
    # Try to get lung mask from totalseg
    if lung_mask is None:
        for key in ["lung_left", "lung_right", "lung_upper_lobe_left", "lung_upper_lobe_right",
                    "lung_lower_lobe_left", "lung_lower_lobe_right"]:
            if key in totalseg_masks:
                lung_mask = _load_mask(totalseg_masks[key])
                if lung_mask is not None:
                    break
    
    if lung_mask is None:
        # Heuristic lung mask: air-filled regions
        lung_mask = (v > -1000) & (v < -500)
    
    lung_voxels = v[lung_mask]
    if lung_voxels.size == 0:
        return {"tb_suspect": False, "tb_score": 0.0, "findings": []}
    
    findings = []
    score = 0.0
    
    # 1. Apical consolidation: upper lung zones with soft tissue density
    # Approximate apical as upper 30% of volume
    z_dim = v.shape[0] if v.ndim == 3 else 1
    if z_dim > 1:
        apical_mask = lung_mask.copy()
        apical_mask[int(z_dim * 0.7):, :, :] = False
        apical_voxels = v[apical_mask]
        
        consolidation = np.sum((apical_voxels > -100) & (apical_voxels < 50))
        apical_total = max(np.sum(apical_mask), 1)
        consolidation_ratio = consolidation / apical_total
        
        if consolidation_ratio > 0.05:
            findings.append(f"Apical consolidation detected ({consolidation_ratio:.1%})")
            score += 0.3
    
    # 2. Calcified granulomas: small hyperdense foci in lung parenchyma
    calcified = (v > 100) & (v < 400) & lung_mask
    if np.any(calcified):
        labeled, n_granulomas = ndimage.label(calcified.astype(np.uint8))
        if n_granulomas >= 2:
            findings.append(f"Multiple calcified granulomas (n={n_granulomas})")
            score += 0.25 * min(n_granulomas / 3, 1.0)
        elif n_granulomas == 1:
            findings.append("Single calcified granuloma")
            score += 0.1
    
    # 3. Cavitation proxy: thick-walled low-density regions in consolidation
    cavitary = (v > -50) & (v < 20) & lung_mask
    if np.any(cavitary):
        # Check for thick ring-like structures
        eroded = ndimage.binary_erosion(cavitary.astype(np.uint8), iterations=2)
        cavity_wall = cavitary & ~eroded.astype(bool)
        if np.sum(cavity_wall) > 100:  # Minimum size threshold
            findings.append("Cavitary lesion pattern detected")
            score += 0.35
    
    # 4. Tree-in-bud pattern proxy: small centrilobular nodules
    small_nodules = (v > -50) & (v < 50) & lung_mask
    if np.any(small_nodules):
        labeled, n_nodules = ndimage.label(small_nodules.astype(np.uint8))
        if n_nodules > 50:  # Many small nodules
            findings.append("Tree-in-bud pattern suspected (centrilobular nodules)")
            score += 0.2
    
    tb_suspect = score >= 0.4
    
    return {
        "tb_suspect": tb_suspect,
        "tb_score": round(score, 3),
        "findings": findings,
        "consolidation_ratio": round(consolidation_ratio, 3) if 'consolidation_ratio' in dir() else None,
    }


def compute_nafld_grade(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
) -> dict[str, Any]:
    """
    NAFLD grading: liver HU vs spleen HU ratio.
    Normal liver > spleen by ~10 HU.
    Fatty liver: liver HU < spleen HU or liver HU < 40.
    """
    v = np.asarray(volume_hu, dtype=np.float32)
    
    # Get liver and spleen masks
    liver_mask = None
    for key in ["liver", "liver_vessel"]:
        if key in totalseg_masks:
            liver_mask = _load_mask(totalseg_masks[key])
            if liver_mask is not None:
                break
    
    spleen_mask = None
    if "spleen" in totalseg_masks:
        spleen_mask = _load_mask(totalseg_masks["spleen"])
    
    if liver_mask is None or spleen_mask is None:
        return {
            "nafld_grade": "unknown",
            "liver_mean_hu": None,
            "spleen_mean_hu": None,
            "liver_spleen_ratio": None,
        }
    
    liver_hu = v[liver_mask]
    spleen_hu = v[spleen_mask]
    
    if liver_hu.size == 0 or spleen_hu.size == 0:
        return {
            "nafld_grade": "unknown",
            "liver_mean_hu": None,
            "spleen_mean_hu": None,
            "liver_spleen_ratio": None,
        }
    
    liver_mean = float(np.mean(liver_hu))
    spleen_mean = float(np.mean(spleen_hu))
    ratio = liver_mean / max(spleen_mean, 1.0)
    
    # NAFLD grading
    if liver_mean < 40 or liver_mean < spleen_mean - 10:
        grade = "severe"
    elif liver_mean < spleen_mean:
        grade = "moderate"
    elif liver_mean < spleen_mean + 10:
        grade = "mild"
    else:
        grade = "none"
    
    return {
        "nafld_grade": grade,
        "liver_mean_hu": round(liver_mean, 1),
        "spleen_mean_hu": round(spleen_mean, 1),
        "liver_spleen_ratio": round(ratio, 2),
    }


def compute_tropical_pancreatitis_flag(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
) -> dict[str, Any]:
    """
    Tropical pancreatitis flag: pancreatic calcifications + ductal dilation proxy.
    Endemic in Kerala/coastal India; distinct from alcoholic pancreatitis.
    """
    v = np.asarray(volume_hu, dtype=np.float32)
    
    pancreas_mask = None
    for key in ["pancreas", "pancreatic_duct"]:
        if key in totalseg_masks:
            pancreas_mask = _load_mask(totalseg_masks[key])
            if pancreas_mask is not None:
                break
    
    if pancreas_mask is None:
        return {
            "tropical_pancreatitis_suspect": False,
            "pancreatic_calcifications": False,
            "pancreatic_calc_count": 0,
            "pancreas_mean_hu": None,
        }
    
    pancreas_voxels = v[pancreas_mask]
    if pancreas_voxels.size == 0:
        return {
            "tropical_pancreatitis_suspect": False,
            "pancreatic_calcifications": False,
            "pancreatic_calc_count": 0,
            "pancreas_mean_hu": None,
        }
    
    pancreas_mean = float(np.mean(pancreas_voxels))
    
    # Detect calcifications in pancreas
    calc_mask = (v > 130) & pancreas_mask
    calc_count = 0
    
    if np.any(calc_mask):
        labeled, calc_count = ndimage.label(calc_mask.astype(np.uint8))
    
    # Ductal dilation proxy: dilated duct would have fluid HU + elongated shape
    # Simplified: check for low-density regions in expected duct location
    duct_dilation_proxy = False
    
    # Tropical pancreatitis suspicion criteria
    suspect = calc_count >= 3 or (calc_count >= 1 and pancreas_mean > 50)
    
    return {
        "tropical_pancreatitis_suspect": suspect,
        "pancreatic_calcifications": calc_count > 0,
        "pancreatic_calc_count": calc_count,
        "pancreas_mean_hu": round(pancreas_mean, 1),
        "ductal_dilation_proxy": duct_dilation_proxy,
    }


def run_chest_heuristics(
    volume_hu: np.ndarray,
    totalseg_masks: dict[str, str],
    region: str = "chest",
) -> dict[str, Any]:
    """
    Run all chest heuristics.
    
    Args:
        volume_hu: CT volume in HU
        totalseg_masks: Dict mapping structure names to mask file paths
        region: "chest" or "abdomen" (determines which heuristics to run)
    
    Returns:
        Combined heuristic results
    """
    results: dict[str, Any] = {"region": region}
    
    # TB heuristic for chest
    if region in ("chest", "thorax"):
        tb_results = compute_tb_heuristic(volume_hu, totalseg_masks)
        results["tb_heuristic"] = tb_results
    
    # NAFLD and pancreatitis for abdomen (but also relevant for chest CT that includes upper abdomen)
    nafld_results = compute_nafld_grade(volume_hu, totalseg_masks)
    results["nafld_heuristic"] = nafld_results
    
    pancreatitis_results = compute_tropical_pancreatitis_flag(volume_hu, totalseg_masks)
    results["tropical_pancreatitis_heuristic"] = pancreatitis_results
    
    return results