"""
Manthana — shared mobile / camera / gallery image intake.

Single entry point for raw bytes before inference. Handles format detection, HEIC, WebP,
EXIF orientation, metadata stripping (DPDP), and optional quality warnings.

CXR (TorchXRayVision): do not duplicate tensor math here — save sanitized pixels to a temp
file and let txrv_utils.txrv_tensor_from_filepath() run the canonical preprocess
(skimage + xrv.datasets.normalize + XRayCenterCrop + XRayResizer).
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("manthana.image_intake")

# ─── Format detection ────────────────────────────────────────────────────────


def _detect_format(raw: bytes) -> str:
    """Detect image format from magic bytes — do not trust file extension alone."""
    if len(raw) < 12:
        return "unknown"
    if raw[:4] == b"\x00\x00\x00\x1c" or b"ftyp" in raw[:12]:
        return "heic"
    if raw[:4] == b"RIFF" and len(raw) >= 12 and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:2] == b"\xff\xd8":
        return "jpeg"
    if raw[:4] == b"\x89PNG":
        return "png"
    if raw[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if raw[:4] == b"%PDF":
        return "pdf"
    return "unknown"


def _fix_exif_rotation(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def _heic_to_pil(raw: bytes) -> Image.Image:
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except ImportError as e:
        raise RuntimeError(
            "HEIC images require pillow-heif. Install: pip install pillow-heif>=0.13"
        ) from e
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _strip_metadata(img: Image.Image) -> Image.Image:
    """Drop EXIF/XMP by round-tripping pixel data (DPDP-style metadata minimization)."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return Image.fromarray(np.asarray(img))


def raw_to_pil(raw: bytes) -> Image.Image:
    """
    Decode mobile bytes to PIL. Applies EXIF transpose and strips metadata.
    Raises ValueError for PDF (extract images upstream).
    """
    fmt = _detect_format(raw)
    if fmt == "pdf":
        raise ValueError(
            "PDF uploads are not decoded here — extract embedded images first "
            "(e.g. PyMuPDF / pdf2image)."
        )
    if fmt == "heic":
        img = _heic_to_pil(raw)
    else:
        img = Image.open(io.BytesIO(raw))
    img = _fix_exif_rotation(img)
    return _strip_metadata(img)


def _quality_check(img: Image.Image, modality: str) -> dict[str, Any]:
    w, h = img.size
    warnings: list[str] = []

    if w < 100 or h < 100:
        warnings.append(
            f"Very low resolution ({w}×{h}). Model accuracy will be reduced. "
            "Recommend at least 512×512 for clinical use when possible."
        )
    elif w < 512 or h < 512:
        warnings.append(
            f"Low resolution ({w}×{h}). Results may be less accurate than "
            "standard radiology acquisition."
        )

    try:
        import cv2

        arr = np.array(img.convert("L"))
        blur_score = float(cv2.Laplacian(arr, cv2.CV_64F).var())
        if blur_score < 50.0:
            warnings.append(
                f"Image appears blurry (sharpness score: {blur_score:.1f}). "
                "Camera shake or focus issues reduce diagnostic utility."
            )
    except Exception:
        pass

    arr = np.array(img.convert("L"), dtype=np.float64)
    mean_b = float(arr.mean())
    if mean_b < 20.0:
        warnings.append("Image is very dark — check lighting when capturing.")
    if mean_b > 235.0:
        warnings.append("Image may be overexposed — avoid direct glare on the film/screen.")

    return {
        "resolution": f"{w}×{h}",
        "warnings": warnings,
        "quality_ok": len(warnings) == 0,
        "modality_hint": modality,
    }


# ─── Per-modality processors (arrays / bytes for non-file pipelines) ─────────

Processor = Callable[[Image.Image], Any]


def _processor_xray(_pil: Image.Image) -> None:
    """CXR uses filepath + txrv_utils — no duplicate tensor here."""
    return None


def _normalize_for_virchow(img: Image.Image, patch_size: int = 224) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_rgb = img.convert("RGB").resize((patch_size, patch_size), Image.LANCZOS)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


def _normalize_for_mirai(img: Image.Image) -> np.ndarray:
    img_gray = img.convert("L").resize((2048, 1664), Image.LANCZOS)
    arr = np.array(img_gray, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 65535.0
    return arr[np.newaxis, ...]


def _normalize_for_claude(img: Image.Image, max_bytes: int = 4_500_000) -> bytes:
    img_rgb = img.convert("RGB")
    max_dim = 2048
    w, h = img_rgb.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img_rgb = img_rgb.resize((int(w * s), int(h * s)), Image.LANCZOS)
    quality = 92
    while quality >= 60:
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=quality, optimize=True)
        raw = buf.getvalue()
        if len(raw) <= max_bytes:
            return raw
        quality -= 8
    img_rgb = img_rgb.resize((img_rgb.width // 2, img_rgb.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=75, optimize=True)
    return buf.getvalue()


MODALITY_PROCESSORS: dict[str, Processor] = {
    "chest_xray": _processor_xray,
    "xray": _processor_xray,
    "pathology": _normalize_for_virchow,
    "cytology": _normalize_for_virchow,
    "mammography": _normalize_for_mirai,
    "dermatology": lambda im: _normalize_for_claude(im),
    "oral_cancer": lambda im: _normalize_for_claude(im),
    "lab_report": lambda im: _normalize_for_claude(im),
    "ecg": lambda im: _normalize_for_claude(im),
    "brain_mri": _normalize_for_virchow,
    "spine_neuro": _normalize_for_virchow,
}


def intake_pil_to_temp_path(
    pil: Image.Image,
    directory: str,
    filename_stem: str,
    *,
    prefer_grayscale: bool = True,
) -> str:
    """
    Write sanitized PIL to a lossless PNG on disk for skimage/torchxrayvision loaders.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    out = Path(directory) / f"{filename_stem}.png"
    face = pil.convert("L") if prefer_grayscale else pil.convert("RGB")
    face.save(str(out), format="PNG", optimize=True)
    return str(out)


def pil_to_jpeg_b64(pil: Image.Image, quality: int = 92) -> str:
    """JPEG bytes as base64 for vision LLM calls."""
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def normalize_for_model(
    raw: bytes,
    modality: str,
) -> dict[str, Any]:
    """
    Universal intake. For CXR/xray, ``array`` is None — save ``pil`` via
    :func:`intake_pil_to_temp_path` and pass that path to ``run_pipeline``.

    Returns:
        array: np.ndarray, bytes (JPEG), or None (use temp file + existing pipelines)
        pil: RGB or L PIL image (sanitized)
        quality: resolution / warnings / quality_ok
        format: detected magic-bytes format
        size_bytes: len(raw)
    """
    fmt = _detect_format(raw)
    pil = raw_to_pil(raw)
    quality = _quality_check(pil, modality)

    key = modality.strip().lower()
    processor = MODALITY_PROCESSORS.get(key)
    if processor is None:
        raise ValueError(f"Unknown modality for image intake: {modality!r}")

    try:
        processed: Any = processor(pil)
    except Exception as e:
        raise RuntimeError(f"Normalization failed for {modality}: {e}") from e

    return {
        "array": processed,
        "pil": pil,
        "quality": quality,
        "format": fmt,
        "size_bytes": len(raw),
    }


def normalize_b64_for_model(image_b64: str, modality: str) -> dict[str, Any]:
    raw = base64.b64decode(image_b64)
    return normalize_for_model(raw, modality)


def merge_image_quality_into_result(result: dict, quality: dict[str, Any]) -> None:
    """Attach ``structures['image_quality']`` for API responses (never blocks inference)."""
    st = result.get("structures")
    if isinstance(st, dict):
        st["image_quality"] = quality
        result["structures"] = st
    elif isinstance(st, list):
        result["structures"] = {
            "legacy_structures_list": st,
            "image_quality": quality,
        }
    else:
        result["structures"] = {"image_quality": quality}
