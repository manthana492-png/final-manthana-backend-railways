"""India-focused NCCT heuristics: midline shift proxy, NCC calcifications, hydrocephalus flag."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy import ndimage


def _brain_mask_from_hu(vol: np.ndarray) -> np.ndarray:
    """Loose brain parenchyma mask (HU window)."""
    v = np.asarray(vol, dtype=np.float32)
    return (v > 15.0) & (v < 100.0)


def compute_midline_shift_mm(
    volume_hu: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float] | None = None,
) -> float | None:
    """
    Bilateral hemisphere center-of-mass delta along R-L axis (assumed last dim = X).
    Returns mm if spacing known, else voxel units as float (caller may scale).
    """
    v = np.asarray(volume_hu, dtype=np.float32)
    if v.ndim == 2:
        v = v[np.newaxis, ...]
    if v.ndim != 3:
        return None
    z, y, x = v.shape
    mid = x // 2
    mask = _brain_mask_from_hu(v)
    if not np.any(mask):
        return None

    def com_x(m: np.ndarray) -> float:
        idx = np.where(m)
        if idx[0].size == 0:
            return float(mid)
        weights = np.ones_like(idx[2], dtype=np.float64)
        return float(np.average(idx[2], weights=weights))

    left = mask[:, :, :mid]
    right = mask[:, :, mid:]
    cx_l = com_x(left)
    cx_r = com_x(right) + mid
    shift_vox = abs(cx_r - cx_l) * 0.5
    if voxel_spacing_mm and len(voxel_spacing_mm) >= 3:
        sx = float(voxel_spacing_mm[2])
        return round(float(shift_vox * sx), 2)
    return round(float(shift_vox), 2)


def detect_ncc_calcifications(
    volume_hu: np.ndarray,
    hu_min: float = 80.0,
    hu_max: float = 300.0,
    max_diameter_mm: float = 10.0,
    voxel_mm: float = 1.0,
) -> tuple[int, float]:
    """Count connected hyperdense foci in parenchyma range (NCC/TB calc differential hint)."""
    v = np.asarray(volume_hu, dtype=np.float32)
    brain = _brain_mask_from_hu(v)
    calc = brain & (v >= hu_min) & (v <= hu_max)
    if not np.any(calc):
        return 0, 0.0
    labeled, n = ndimage.label(calc.astype(np.uint8))
    count = 0
    max_d_vox = max(1, int(max_diameter_mm / max(voxel_mm, 0.1)))
    for i in range(1, n + 1):
        coords = np.where(labeled == i)
        if coords[0].size == 0:
            continue
        dz = coords[0].max() - coords[0].min() + 1
        dy = coords[1].max() - coords[1].min() + 1
        dx = coords[2].max() - coords[2].min() + 1
        diam = max(dz, dy, dx) * voxel_mm
        if diam <= max_diameter_mm + voxel_mm:
            count += 1
    suspect = 1.0 if count >= 2 else (0.4 if count == 1 else 0.0)
    return count, suspect


def classify_subdural_chronicity(
    volume_hu: np.ndarray,
    sdh_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    If sdh_mask provided, mean HU in mask. Else rough crescent search skipped — return unknown.
    """
    if sdh_mask is None or not np.any(sdh_mask):
        return {"chronic_sdh_flag": False, "acute_sdh_flag": False, "sdh_mean_hu": None}
    v = np.asarray(volume_hu, dtype=np.float32)
    m = np.asarray(sdh_mask, dtype=bool)
    mean_hu = float(np.mean(v[m]))
    chronic = 20.0 <= mean_hu <= 40.0
    acute = mean_hu > 60.0
    return {
        "chronic_sdh_flag": chronic,
        "acute_sdh_flag": acute,
        "sdh_mean_hu": round(mean_hu, 1),
    }


def compute_hydrocephalus_flag(ventricle_volume_ml: float | None, threshold_ml: float = 30.0) -> bool:
    if ventricle_volume_ml is None:
        return False
    return float(ventricle_volume_ml) > threshold_ml


def compute_calcification_burden(volume_hu: np.ndarray, hu_min: float = 100.0) -> float:
    """Fraction of intracranial voxels above hu_min (coarse calc burden)."""
    v = np.asarray(volume_hu, dtype=np.float32)
    brain = _brain_mask_from_hu(v) | ((v > -50) & (v < 200))
    if not np.any(brain):
        return 0.0
    calc = (v >= hu_min) & brain
    return round(float(np.sum(calc) / max(np.sum(brain), 1)), 4)


def run_ct_brain_heuristics(
    volume_hu: np.ndarray,
    meta: dict,
    ventricle_volume_ml: float | None,
    enabled_ncc: bool = True,
    enabled_midline: bool = True,
) -> dict[str, Any]:
    ps = meta.get("pixel_spacing") if isinstance(meta, dict) else None
    st = meta.get("slice_thickness") if isinstance(meta, dict) else None
    vx = 1.0
    if isinstance(ps, (list, tuple)) and len(ps) >= 2:
        try:
            vx = (float(ps[0]) + float(ps[1])) / 2.0
        except (TypeError, ValueError):
            pass
    try:
        st_f = float(st) if st is not None else 1.0
    except (TypeError, ValueError):
        st_f = 1.0
    py = float(ps[0]) if isinstance(ps, (list, tuple)) and len(ps) > 0 else 1.0
    px = float(ps[1]) if isinstance(ps, (list, tuple)) and len(ps) > 1 else 1.0
    spacing = (st_f, py, px)

    out: dict[str, Any] = {}
    if enabled_midline:
        ms = compute_midline_shift_mm(volume_hu, spacing)
        if ms is not None:
            out["midline_shift_mm"] = ms
    if enabled_ncc:
        ncc_n, ncc_sus = detect_ncc_calcifications(volume_hu, voxel_mm=vx)
        out["calcification_foci_count"] = ncc_n
        out["ncc_suspect_score"] = round(ncc_sus, 3)
        out["ncc_suspect"] = ncc_n >= 2 or ncc_sus >= 0.9
    out["calcification_burden"] = compute_calcification_burden(volume_hu)
    out["hydrocephalus_flag"] = compute_hydrocephalus_flag(
        ventricle_volume_ml,
        float(os.getenv("CT_BRAIN_HYDROCEPHALUS_ML_THRESHOLD", "30") or "30"),
    )
    sdh = classify_subdural_chronicity(volume_hu, None)
    out.update({k: v for k, v in sdh.items() if k != "sdh_mean_hu" or v is not None})
    return out
