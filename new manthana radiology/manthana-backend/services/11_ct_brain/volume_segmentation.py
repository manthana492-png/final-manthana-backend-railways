"""
Optional volumetric segmentation for CT brain (TorchScript nnU-Net-style bundle).
Falls back to HU-based ventricle proxy when CT_BRAIN_SEGMENTATION_MODEL_PATH unset.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.ct_brain.seg")

_seg_model: Any = None


def voxel_volume_mm3_from_meta(meta: dict, volume_shape: tuple[int, ...]) -> float:
    """Approximate mm³ per voxel from DICOM meta or NIfTI affine."""
    if not isinstance(meta, dict):
        return 1.0
    ps = meta.get("pixel_spacing")
    st = meta.get("slice_thickness")
    if isinstance(ps, (list, tuple)) and len(ps) >= 2 and st is not None:
        try:
            return float(ps[0]) * float(ps[1]) * float(st)
        except (TypeError, ValueError):
            pass
    aff = meta.get("affine")
    if aff is not None:
        try:
            a = np.asarray(aff, dtype=np.float64)
            if a.shape == (4, 4):
                return float(abs(np.linalg.det(a[:3, :3])))
        except Exception:
            pass
    # default 1mm isotropic guess
    z, y, x = volume_shape[:3]
    return 1.0


def _torch_device():
    import torch

    if os.getenv("CT_BRAIN_DEVICE", "").strip().lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_seg_model() -> Any | None:
    global _seg_model
    path = (os.environ.get("CT_BRAIN_SEGMENTATION_MODEL_PATH") or "").strip()
    if not path or not os.path.isfile(path):
        return None
    if _seg_model is not None:
        return _seg_model
    import torch

    try:
        _seg_model = torch.jit.load(path, map_location=_torch_device())
        _seg_model.eval()
        logger.info("Loaded CT Brain segmentation TorchScript from %s", path)
    except Exception as e:
        logger.warning("CT Brain segmentation load failed: %s", e)
        return None
    return _seg_model


def _hu_proxy_ventricle_volume_ml(volume_hu: np.ndarray, voxel_mm3: float) -> float:
    """CSF-like voxels: HU roughly 0–25 inside brain window context (rough proxy)."""
    v = np.asarray(volume_hu, dtype=np.float32)
    # ventricular CSF often ~0-20 HU on NCCT
    mask = (v >= -5.0) & (v <= 25.0)
    # exclude air
    mask &= v > -200
    n = int(np.sum(mask))
    return round(n * voxel_mm3 / 1000.0, 2)


def _hu_proxy_hemorrhage_volume_ml(volume_hu: np.ndarray, voxel_mm3: float) -> float:
    """Hyperdense acute blood proxy HU > 40."""
    v = np.asarray(volume_hu, dtype=np.float32)
    mask = v > 40.0
    n = int(np.sum(mask))
    return round(n * voxel_mm3 / 1000.0, 2)


def run_volume_analysis(
    volume_hu: np.ndarray,
    meta: dict,
    tensor_5d: Any | None = None,
) -> dict[str, Any]:
    """
    Returns hemorrhage_volume_ml, ventricle_volume_ml, source, optional masks note.
    If TorchScript seg model is set, caller may extend; current impl uses HU proxies when no model.
    """
    v = np.asarray(volume_hu, dtype=np.float32)
    if v.ndim == 2:
        v = v[np.newaxis, ...]
    shape = v.shape
    voxel_mm3 = voxel_volume_mm3_from_meta(meta if isinstance(meta, dict) else {}, shape)

    out: dict[str, Any] = {
        "hemorrhage_volume_ml": None,
        "ventricle_volume_ml": None,
        "segmentation_source": "none",
    }

    m = _load_seg_model()
    if m is not None and tensor_5d is not None:
        try:
            import torch

            dev = _torch_device()
            t = tensor_5d.to(dev)
            with torch.no_grad():
                seg = m(t)
            # Expect multi-channel logits (1,C,D,H,W) — take argmax channel 1 as blood if C>=2
            if isinstance(seg, torch.Tensor) and seg.dim() == 5 and seg.shape[1] >= 2:
                pred = torch.argmax(seg, dim=1).squeeze().cpu().numpy()
                blood = float(np.sum(pred == 1)) * voxel_mm3 / 1000.0
                vent = float(np.sum(pred == 2)) * voxel_mm3 / 1000.0 if seg.shape[1] > 2 else None
                out["hemorrhage_volume_ml"] = round(blood, 2)
                if vent is not None:
                    out["ventricle_volume_ml"] = round(vent, 2)
                out["segmentation_source"] = "torchscript"
                return out
        except Exception as e:
            logger.warning("CT Brain TorchScript segmentation failed, using HU proxy: %s", e)

    out["ventricle_volume_ml"] = _hu_proxy_ventricle_volume_ml(v, voxel_mm3)
    out["hemorrhage_volume_ml"] = _hu_proxy_hemorrhage_volume_ml(v, voxel_mm3)
    out["segmentation_source"] = "hu_proxy"
    return out


def reset_seg_cache_for_tests() -> None:
    global _seg_model
    _seg_model = None
