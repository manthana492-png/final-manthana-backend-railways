"""CT volume loading: single file, NIfTI, or DICOM series directory."""

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np

logger = logging.getLogger("manthana.ct_loader")


def load_ct_volume(
    filepath: str,
    series_dir: str | None = None,
) -> tuple[np.ndarray, dict, bool]:
    """
    Load CT volume for inference.

    If series_dir is set and is a directory, load full DICOM series from there.
    Otherwise load filepath (NIfTI or single DICOM).

    Returns:
        (volume, metadata, series_available)
    """
    if series_dir and os.path.isdir(series_dir):
        from preprocessing.dicom_utils import read_dicom_series

        vol, meta = read_dicom_series(series_dir)
        return vol, meta, True

    if filepath and os.path.isdir(filepath):
        from preprocessing.dicom_utils import read_dicom_series

        vol, meta = read_dicom_series(filepath)
        return vol, meta, True

    ext = filepath.lower().split(".")[-1]
    if ext in ("nii", "gz"):
        from preprocessing.nifti_utils import read_nifti

        v, aff = read_nifti(filepath)
        return v, {"affine": aff}, False

    if ext == "dcm":
        from preprocessing.dicom_utils import read_dicom

        a, meta = read_dicom(filepath)
        return a, meta, False

    from preprocessing.image_utils import load_image, to_grayscale

    return to_grayscale(load_image(filepath)), {}, False


def is_degraded_single_slice(volume: np.ndarray) -> bool:
    """True if volume is 2D or effectively one slice."""
    v = np.asarray(volume)
    if v.ndim < 3:
        return True
    if v.ndim == 3 and min(v.shape) == 1:
        return True
    # (1, H, W) stacked
    if v.shape[0] == 1 or v.shape[2] == 1:
        return True
    return False
