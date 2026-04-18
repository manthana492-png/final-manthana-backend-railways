from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def voxel_spacing_from_meta(meta: dict[str, Any]) -> tuple[float, float, float]:
    spacing = meta.get("spacing") or meta.get("pixdim")
    if isinstance(spacing, (list, tuple)) and len(spacing) >= 3:
        try:
            sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
            if sx > 0 and sy > 0 and sz > 0:
                return sx, sy, sz
        except (TypeError, ValueError):
            pass

    try:
        hdr = meta.get("nifti_header")
        if hdr is not None and hasattr(hdr, "get_zooms"):
            z = hdr.get_zooms()
            if len(z) >= 3:
                return float(z[0]), float(z[1]), float(z[2])
    except Exception:
        pass

    return 1.0, 1.0, 1.0


def volume_ml_for_class(
    segmentation_mask: np.ndarray,
    class_index: int,
    spacing_xyz: tuple[float, float, float],
) -> float:
    voxel_count = float((segmentation_mask == class_index).sum())
    voxel_volume_ml = (spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]) / 1000.0
    return round(voxel_count * voxel_volume_ml, 3)


def build_affine_from_spacing(spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Diagonal mm spacing → 4x4 NIfTI affine (LPS-style diagonal; sufficient for downstream I/O)."""
    a = np.eye(4, dtype=np.float32)
    a[0, 0] = float(spacing_xyz[0])
    a[1, 1] = float(spacing_xyz[1])
    a[2, 2] = float(spacing_xyz[2])
    return a


def export_ct_volume_nifti_gz(
    volume: np.ndarray,
    output_path: Path,
    spacing_xyz: tuple[float, float, float],
) -> Path:
    """Write CT HU volume as compressed NIfTI for NVIDIA VISTA NIM (expects URL-accessible NIfTI)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.asarray(volume, dtype=np.float32)
    affine = build_affine_from_spacing(spacing_xyz)
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, str(output_path))
    return output_path


def export_segmentation_nifti(
    segmentation_mask: np.ndarray,
    output_path: Path,
    *,
    affine: np.ndarray | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(segmentation_mask.astype(np.uint8), affine)
    nib.save(img, str(output_path))
    return output_path

