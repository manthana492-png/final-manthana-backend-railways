"""Manthana — TotalSegmentator Runner (Shared)
Downloads model weights on first run to GPU.
Used by: abdominal_ct, cardiac_ct, brain_mri, spine_neuro.

Supported tasks:
  CT:  "total" (117 structs), "heartchambers", "vertebrae_body", "lung_vessels"
  MRI: "total_mr" (80 structs)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

from totalseg_label_maps import map_organ_key
from totalseg_idle import ensure_totalseg_idle_reaper_started, touch_totalseg_gpu_activity

logger = logging.getLogger("manthana.totalseg_runner")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")


def _totalseg_work_root() -> str:
    """Prefer MODEL_DIR when present; otherwise a writable temp dir (no /models in dev)."""
    if os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    base = os.environ.get("TOTALSEG_WORK_DIR") or os.path.join(
            tempfile.gettempdir(), "manthana_totalseg")
    os.makedirs(base, exist_ok=True)
    return base


def _ensure_nifti_path(
    input_path: Union[str, Path, np.ndarray],
    affine: np.ndarray | None = None,
) -> tuple[str, bool]:
    """Return path to NIfTI and whether a temp dir was created."""
    if isinstance(input_path, (str, Path)):
        return str(input_path), False

    try:
        import nibabel as nib
    except ImportError as e:
        raise RuntimeError("nibabel is required for volume→NIfTI conversion") from e

    aff = affine if affine is not None else np.eye(4)
    tmp = tempfile.mkdtemp(prefix="totalseg_in_", dir=_totalseg_work_root())
    path = os.path.join(tmp, "input.nii.gz")
    img = nib.Nifti1Image(np.asarray(input_path, dtype=np.float32), aff)
    nib.save(img, path)
    return path, True


def run_totalseg(
    input_path: Union[str, Path, np.ndarray],
    output_dir: str | None = None,
    task: str = "total",
    fast: bool = False,
    device: str = "gpu",
    affine: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Run TotalSegmentator and return structure labels + optional mask paths.

    Returns:
        {
          "structure_names": list[str],
          "mask_paths": {name: path},
          "output_dir": str,
        }
    """
    from totalsegmentator.python_api import totalsegmentator

    nii_path, _ = _ensure_nifti_path(input_path, affine=affine)
    out = output_dir or tempfile.mkdtemp(prefix=f"totalseg_{task}_", dir=_totalseg_work_root())
    os.makedirs(out, exist_ok=True)

    ensure_totalseg_idle_reaper_started()
    totalsegmentator(nii_path, out, task=task, fast=fast, device=device)
    touch_totalseg_gpu_activity()

    mask_paths: dict[str, str] = {}
    for f in sorted(Path(out).rglob("*.nii.gz")):
        name = f.name.replace(".nii.gz", "").replace(".nii", "")
        mask_paths[name] = str(f)

    structure_names = sorted(mask_paths.keys())
    if not structure_names:
        logger.warning("TotalSegmentator produced no .nii.gz outputs in %s", out)

    volumes_cm3 = compute_organ_volumes(out, task=task)

    return {
        "structure_names": structure_names,
        "mask_paths": mask_paths,
        "output_dir": out,
        "volumes_cm3": volumes_cm3,
    }


def get_totalseg_version() -> str:
    try:
        import totalsegmentator

        return str(getattr(totalsegmentator, "__version__", "unknown"))
    except Exception:
        return "unknown"


def compute_organ_volumes(output_dir: str, task: str = "total") -> dict[str, float]:
    """Voxel volumes in cm³ from TotalSegmentator mask NIfTI files."""
    import nibabel as nib

    volumes: dict[str, float] = {}
    for f in sorted(Path(output_dir).rglob("*.nii.gz")):
        stem = f.name.replace(".nii.gz", "").replace(".nii", "")
        key = map_organ_key(stem, task)
        if key is None:
            continue
        try:
            img = nib.load(str(f))
            data = img.get_fdata()
            zooms = img.header.get_zooms()[:3]
            voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])
            n = int(np.count_nonzero(data))
            volumes[key] = round(n * voxel_vol_mm3 / 1000.0, 2)
        except Exception as e:
            logger.warning("Volume extraction failed for %s: %s", stem, e)
    return volumes


def estimate_aortic_diameter_mm(output_dir: str) -> dict[str, Any]:
    """
    Max aortic diameter (mm) from TotalSegmentator aorta.nii.gz mask.

    Assumes standard NIfTI axial stacking where slice axis has multiple frames;
    oblique acquisitions may bias diameter. Use axially acquired CT with
    standard DICOM→NIfTI conversion for best results.
    """
    import nibabel as nib

    path = os.path.join(output_dir, "aorta.nii.gz")
    if not os.path.isfile(path):
        return {
            "max_aorta_diameter_mm": None,
            "aaa_detected": None,
            "aaa_risk_flag": None,
            "source": "mask_missing",
        }

    try:
        img = nib.load(path)
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    except Exception as e:
        logger.warning("estimate_aortic_diameter_mm load failed: %s", e)
        return {
            "max_aorta_diameter_mm": None,
            "aaa_detected": None,
            "aaa_risk_flag": None,
            "source": "load_error",
        }

    if data.ndim != 3:
        return {
            "max_aorta_diameter_mm": None,
            "aaa_detected": None,
            "aaa_risk_flag": None,
            "source": "not_3d",
        }

    # Slice axis: prefer last dimension when it has multiple slices (typical LPS axial)
    slice_axis = 2 if data.shape[2] > 1 else (0 if data.shape[0] > 1 else 1)
    max_diameter_mm = 0.0
    for i in range(data.shape[slice_axis]):
        slc = np.take(data, i, axis=slice_axis)
        if slc.sum() < 1e-6:
            continue
        count = int(np.count_nonzero(slc))
        others = [j for j in range(3) if j != slice_axis]
        area_mm2 = count * zooms[others[0]] * zooms[others[1]]
        diameter_mm = 2.0 * np.sqrt(max(area_mm2, 1e-9) / np.pi)
        max_diameter_mm = max(max_diameter_mm, float(diameter_mm))

    if max_diameter_mm <= 0:
        return {
            "max_aorta_diameter_mm": None,
            "aaa_detected": None,
            "aaa_risk_flag": None,
            "source": "empty_mask",
        }

    return {
        "max_aorta_diameter_mm": round(max_diameter_mm, 1),
        "aaa_detected": max_diameter_mm > 30.0,
        "aaa_risk_flag": max_diameter_mm > 50.0,
        "source": "totalseg_aorta_mask",
    }


def structure_list_from_result(result: Mapping[str, Any]) -> list[str]:
    """Normalize runner output to a plain list of structure names for APIs."""
    return list(result.get("structure_names") or [])
