"""CT volume loading: single file, NIfTI, or DICOM series directory."""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("manthana.ct_loader")


def _use_monai_ct_loader() -> bool:
    return os.environ.get("MANTHANA_USE_MONAI_CT_LOADER", "").lower() in (
        "1",
        "true",
        "yes",
    )


def load_ct_volume_monai(
    filepath: str,
    series_dir: str | None = None,
) -> tuple[np.ndarray, dict, bool]:
    """
    Load CT volume using MONAI LoadImage (RAS orientation, consistent metadata).

    Use for Modal/GPU services when ``monai`` is installed. Set
    ``MANTHANA_USE_MONAI_CT_LOADER=1`` to route ``load_ct_volume`` here.

    Returns the same tuple shape as ``load_ct_volume``.
    """
    from monai.transforms import LoadImage

    path = filepath
    series_avail = False
    if series_dir and os.path.isdir(series_dir):
        path = series_dir
        series_avail = True
    elif filepath and os.path.isdir(filepath):
        path = filepath
        series_avail = True

    loader = LoadImage(image_only=False, ensure_channel_first=True)
    data, meta = loader(path)
    arr = np.squeeze(np.asarray(data))

    meta_dict: dict = {}
    if hasattr(meta, "keys"):
        try:
            meta_dict = dict(meta)
        except Exception:
            meta_dict = {"monai_meta": str(meta)}
    else:
        meta_dict = {"monai_meta": str(meta)}

    meta_dict["ct_loader"] = "monai"
    return arr.astype(np.float32, copy=False), meta_dict, series_avail


def _load_series_dir_or_film(directory: str) -> tuple[np.ndarray, dict, bool]:
    """Load DICOM series or 4+ film photos (PNG/JPG/…) from a directory."""
    from preprocessing.film_photo_loader import (
        discover_raster_files,
        is_film_photo_input,
        load_film_photos_as_volume,
    )
    from preprocessing.dicom_utils import read_dicom_series

    if is_film_photo_input(directory):
        paths = discover_raster_files(directory)
        vol, meta = load_film_photos_as_volume(paths)
        return vol, meta, True
    vol, meta = read_dicom_series(directory)
    return vol, meta, True


def load_ct_volume(
    filepath: str,
    series_dir: str | None = None,
) -> tuple[np.ndarray, dict, bool]:
    """
    Load CT volume for inference.

    If series_dir is set and is a directory, load full DICOM series from there.
    Otherwise load filepath (NIfTI or single DICOM).

    When ``MANTHANA_USE_MONAI_CT_LOADER=1`` and MONAI is installed, uses
    :func:`load_ct_volume_monai` (falls back to legacy loaders on error).

    Returns:
        (volume, metadata, series_available)
    """
    if _use_monai_ct_loader():
        dir_candidate = series_dir if (series_dir and os.path.isdir(series_dir)) else filepath
        if dir_candidate and os.path.isdir(dir_candidate):
            from preprocessing.film_photo_loader import is_film_photo_input

            if is_film_photo_input(dir_candidate):
                pass  # use legacy film / DICOM series path below
            else:
                try:
                    return load_ct_volume_monai(filepath, series_dir=series_dir)
                except Exception as e:
                    logger.warning("MONAI CT loader failed, using legacy path: %s", e)
        else:
            try:
                return load_ct_volume_monai(filepath, series_dir=series_dir)
            except Exception as e:
                logger.warning("MONAI CT loader failed, using legacy path: %s", e)

    if series_dir and os.path.isdir(series_dir):
        return _load_series_dir_or_film(series_dir)

    if filepath and os.path.isdir(filepath):
        return _load_series_dir_or_film(filepath)

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
