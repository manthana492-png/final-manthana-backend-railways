"""
Manthana — CT/MRI film photo loader.

Stacks 4+ mobile phone photos of printed CT/MRI film sheets into a pseudo-3D volume
for degraded / triage pipelines. Not a substitute for DICOM.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.film_photo_loader")

RASTER_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
)
DICOM_SUFFIXES = (".dcm", ".dic", ".dicom")

MIN_FILM_SLICES = 4
TARGET_HW = 512


def _natural_sort_key(path: str) -> tuple[list[int | str], str]:
    """Split filename into numeric chunks for natural ordering."""
    base = os.path.basename(path).lower()
    parts = re.split(r"(\d+)", base)
    key: list[int | str] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return key, base


def _exif_datetime_from_path(path: str) -> float | None:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            for tag_id, val in exif.items():
                name = TAGS.get(tag_id, tag_id)
                if name in ("DateTimeOriginal", "DateTime", "DateTimeDigitized") and val:
                    try:
                        dt = datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")
                        return dt.timestamp()
                    except Exception:
                        continue
    except Exception:
        pass
    return None


def discover_raster_files(directory: str) -> list[str]:
    """Return sorted list of absolute paths to raster images in a directory (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out: list[str] = []
    for f in d.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in RASTER_SUFFIXES:
            out.append(str(f.resolve()))
    return out


def count_dicom_files(directory: str) -> int:
    n = 0
    d = Path(directory)
    if not d.is_dir():
        return 0
    for f in d.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in DICOM_SUFFIXES or f.name.lower().endswith(DICOM_SUFFIXES):
            n += 1
    return n


def is_film_photo_input(directory: str) -> bool:
    """
    True if directory looks like a film-photo batch: >=4 raster images and no DICOM files.
    Avoids misclassifying DICOM series folders that happen to contain thumbnails.
    """
    if not directory or not os.path.isdir(directory):
        return False
    if count_dicom_files(directory) > 0:
        return False
    rasters = discover_raster_files(directory)
    return len(rasters) >= MIN_FILM_SLICES


def _crop_dark_borders(gray: np.ndarray, percentile: float = 5.0) -> np.ndarray:
    """Remove outer rows/cols that are mostly black (film edge / phone bezel)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim != 2:
        return gray
    h, w = g.shape
    row_mean = g.mean(axis=1)
    col_mean = g.mean(axis=0)
    thr_r = float(np.percentile(row_mean, percentile))
    thr_c = float(np.percentile(col_mean, percentile))
    # Keep rows/cols above global noise floor
    floor = float(g.min()) + 0.05 * (float(g.max()) - float(g.min()) + 1e-6)
    rmask = row_mean > max(thr_r, floor + 1.0)
    cmask = col_mean > max(thr_c, floor + 1.0)
    if not rmask.any() or not cmask.any():
        return gray
    ys = np.where(rmask)[0]
    xs = np.where(cmask)[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    pad = 2
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    if y1 - y0 < 32 or x1 - x0 < 32:
        return gray
    return g[y0:y1, x0:x1].astype(gray.dtype)


def _process_one_slice(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one film photo -> uint8 grayscale (H,W) after CLAHE + resize."""
    from preprocessing.image_utils import apply_clahe, resize_image, to_grayscale

    try:
        from image_intake import raw_to_pil
    except ImportError:
        from preprocessing.image_utils import load_image

        arr = load_image(path)
        gray = to_grayscale(arr)
        pil_mode = "loaded_path"
    else:
        with open(path, "rb") as f:
            raw = f.read()
        pil = raw_to_pil(raw)
        gray = np.asarray(pil.convert("L"), dtype=np.uint8)
        pil_mode = "intake"

    gray = _crop_dark_borders(gray)
    enhanced = apply_clahe(gray, clip_limit=2.0, tile_size=8)
    resized = resize_image(enhanced, (TARGET_HW, TARGET_HW), keep_aspect=True)
    if resized.ndim == 3:
        resized = to_grayscale(resized)

    # Blur score (Laplacian variance) for quality
    blur_score: float | None = None
    try:
        import cv2

        blur_score = float(cv2.Laplacian(np.asarray(resized, dtype=np.uint8), cv2.CV_64F).var())
    except Exception:
        pass

    mean_b = float(np.asarray(resized, dtype=np.float64).mean())
    meta = {
        "path": path,
        "blur_score": blur_score,
        "mean_brightness": mean_b,
        "decoder": pil_mode,
    }
    return np.asarray(resized, dtype=np.uint8), meta


def _sort_image_paths(paths: list[str]) -> list[str]:
    """Sort by EXIF time when present, else natural filename order."""

    def sort_key(p: str) -> tuple[int, float, tuple[list[int | str], str]]:
        ts = _exif_datetime_from_path(p)
        nat = _natural_sort_key(p)
        # Primary: has exif (0) before no exif (1); then timestamp; then natural name
        has_exif = 0 if ts is not None else 1
        return (has_exif, ts if ts is not None else 0.0, nat)

    return sorted(paths, key=sort_key)


def load_film_photos_as_volume(image_paths: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load and stack film photos into pseudo-3D volume (Z, H, W) float32 in HU-like scale proxy.

    Minimum MIN_FILM_SLICES images required.

    Returns:
        volume: float32 array, shape (N, TARGET_HW, TARGET_HW), scaled roughly 0-2000 for CT-like windowing
        metadata: includes film_photo_mode, per-slice quality, synthetic spacing
    """
    paths = [os.path.abspath(p) for p in image_paths if p and os.path.isfile(p)]
    paths = list(dict.fromkeys(paths))
    if len(paths) < MIN_FILM_SLICES:
        raise ValueError(
            f"Film photo mode requires at least {MIN_FILM_SLICES} images; got {len(paths)}"
        )

    paths = _sort_image_paths(paths)
    slices: list[np.ndarray] = []
    per_slice_meta: list[dict[str, Any]] = []

    for p in paths:
        sl, sm = _process_one_slice(p)
        slices.append(sl)
        per_slice_meta.append(sm)

    stacked = np.stack(slices, axis=0).astype(np.float32)
    # Map 0-255 film photo pixels to a CT-like float range so downstream windowing (e.g. brain) works
    vol = stacked * (2000.0 / 255.0)

    warnings: list[str] = []
    blur_scores = [m.get("blur_score") for m in per_slice_meta if m.get("blur_score") is not None]
    if blur_scores and min(blur_scores) < 50.0:
        warnings.append("One or more film photos appear blurry; results are less reliable.")

    means = [m.get("mean_brightness", 128) for m in per_slice_meta]
    if means and (min(means) < 25 or max(means) > 230):
        warnings.append("Film photos may be under/over-exposed; check lighting when capturing.")

    quality_ok = len(warnings) == 0

    meta: dict[str, Any] = {
        "source": "film_photo",
        "film_photo_mode": True,
        "num_slices": int(vol.shape[0]),
        "volume_shape": list(vol.shape),
        "pixel_spacing": [1.0, 1.0],
        "slice_thickness": 5.0,
        "modality": "",  # caller may set CT vs MR hint
        "per_slice_quality": per_slice_meta,
        "film_photo_quality": {
            "quality_ok": quality_ok,
            "warnings": warnings,
        },
        # Store original paths for potential LLM vision extraction
        "_source_paths": paths,
    }
    logger.info(
        "Loaded film photo volume: %d slices, shape=%s",
        vol.shape[0],
        vol.shape,
    )
    return vol, meta


def extract_representative_slices_for_llm(
    meta: dict[str, Any],
    max_images: int = 10,
    min_quality_threshold: float = 50.0,
) -> list[str]:
    """
    Extract base64-encoded PNG images from film-photo slices for LLM vision.
    
    Strategy:
    - Skip first and last slices (often scouts or incomplete)
    - Select slices with best quality scores (blur, brightness)
    - Evenly distribute across remaining slices to capture anatomical variation
    - Return base64-encoded PNG strings
    
    Args:
        meta: Metadata dict from load_film_photos_as_volume
        max_images: Maximum number of images to return (4-15 recommended for cost control)
        min_quality_threshold: Skip slices with blur_score below this
    
    Returns:
        List of base64-encoded PNG strings for LLM vision input
    """
    paths = meta.get("_source_paths", [])
    per_slice = meta.get("per_slice_quality", [])
    num_slices = len(paths)
    
    if num_slices < MIN_FILM_SLICES or len(per_slice) != num_slices:
        logger.warning("Cannot extract LLM images: invalid meta or missing paths")
        return []
    
    # Build quality-ranked indices (skip first/last as they're often scouts)
    candidates = []
    for i in range(1, num_slices - 1):  # Skip index 0 and last
        quality = per_slice[i] if i < len(per_slice) else {}
        blur = quality.get("blur_score", 0.0) or 0.0
        brightness = quality.get("mean_brightness", 128.0) or 128.0
        
        # Quality score: prefer mid-range brightness (100-180) and low blur
        brightness_score = 1.0 - abs(brightness - 140) / 140.0
        quality_score = (blur / 200.0 if blur else 0.5) * 0.7 + brightness_score * 0.3
        
        if blur >= min_quality_threshold or blur == 0:  # 0 means couldn't compute, assume ok
            candidates.append((i, quality_score, paths[i]))
    
    if not candidates:
        logger.warning("No quality slices found for LLM vision, using middle slice only")
        mid = num_slices // 2
        candidates = [(mid, 0.5, paths[mid])]
    
    # Sort by quality descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Select top candidates but ensure anatomical distribution
    selected_indices = set()
    for idx, _, _ in candidates:
        if len(selected_indices) >= max_images:
            break
        # Add if not too close to already selected (minimum spacing of 1 slice)
        too_close = any(abs(idx - s) < 2 for s in selected_indices)
        if not too_close or len(selected_indices) < 4:  # Allow close ones if we need minimum
            selected_indices.add(idx)
    
    # If we have few selections, fill with evenly distributed slices
    if len(selected_indices) < min(4, num_slices - 2):
        step = max(1, (num_slices - 2) // min(4, max_images))
        for i in range(1, num_slices - 1, step):
            if len(selected_indices) >= max_images:
                break
            selected_indices.add(i)
    
    # Convert to sorted list
    selected = sorted(selected_indices)
    
    # Encode images as base64 PNG
    import base64
    from io import BytesIO
    from PIL import Image
    
    encoded: list[str] = []
    for idx in selected:
        try:
            path = paths[idx]
            with Image.open(path) as img:
                # Convert to grayscale, ensure consistent size
                if img.mode != 'L':
                    img = img.convert('L')
                # Resize if too large (keep max dimension 768 for token efficiency)
                max_dim = 768
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                buf = BytesIO()
                img.save(buf, format='PNG')
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                encoded.append(b64)
        except Exception as e:
            logger.warning("Failed to encode slice %d for LLM: %s", idx, e)
            continue
    
    logger.info("Extracted %d representative film-photo slices for LLM vision", len(encoded))
    return encoded
