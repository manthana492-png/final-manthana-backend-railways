"""Pre-digitization quality hints for ECG photos (blur, resolution)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("manthana.ecg_image_quality")


def assess_ecg_image_quality(
    filepath: str,
    min_short_edge: int = 480,
    blur_variance_min: float = 15.0,
) -> dict[str, Any]:
    """
    Returns ok=False for obviously unusable images; still allows pipeline to proceed
    (hybrid mode) — caller sets structures warnings.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV unavailable — skipping ECG image quality gate")
        return {
            "ok": True,
            "skipped": True,
            "blur_variance": None,
            "short_edge": None,
            "warnings": [],
        }

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "ok": False,
            "blur_variance": None,
            "short_edge": None,
            "warnings": ["Could not read image file."],
        }

    h, w = img.shape[:2]
    short = int(min(h, w))
    blur_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

    warnings: list[str] = []
    ok = True
    if short < min_short_edge:
        ok = False
        warnings.append(
            f"Image is small (short edge {short}px). Prefer at least {min_short_edge}px for reliable digitization."
        )
    if blur_var < blur_variance_min:
        ok = False
        warnings.append(
            "Image appears blurry or low-contrast. Retake with steady hands and even lighting."
        )

    return {
        "ok": ok,
        "blur_variance": blur_var,
        "short_edge": short,
        "warnings": warnings,
    }
