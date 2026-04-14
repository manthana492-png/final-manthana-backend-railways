"""
Best-effort conversion of uploads to JPEG/PNG or text for VLM / LLM.
Optional deps: pydicom, Pillow, pdfplumber, nibabel, opencv — import lazily.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("manthana.file_converter")

MimeData = Tuple[str, str]  # mime, b64


def _pil_to_jpeg_b64(raw: bytes) -> Optional[str]:
    try:
        from PIL import Image  # type: ignore

        im = Image.open(io.BytesIO(raw))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.debug("PIL convert failed: %s", e)
        return None


def convert_upload_to_image_b64(
    raw: bytes,
    filename: str,
    content_type: Optional[str],
) -> Optional[MimeData]:
    """
    Return (mime, base64) suitable for data URL, or None to let caller use raw as text.
    """
    ct = (content_type or "").lower()
    name = (filename or "").lower()

    if ct.startswith("image/") or name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        b64 = base64.b64encode(raw).decode("ascii")
        mime = "image/jpeg"
        if "png" in ct or name.endswith(".png"):
            mime = "image/png"
        elif "webp" in ct or name.endswith(".webp"):
            mime = "image/webp"
        return mime, b64

    if name.endswith(".dcm") or "dicom" in ct:
        try:
            import pydicom  # type: ignore
            from PIL import Image  # type: ignore

            ds = pydicom.dcmread(io.BytesIO(raw), force=True)
            arr = ds.pixel_array  # type: ignore[attr-defined]
            im = Image.fromarray(arr.astype("uint8") if arr.ndim == 2 else arr[:, :, 0])
            if im.mode != "RGB":
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
            return "image/jpeg", base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.warning("DICOM conversion failed: %s", e)
            return None

    if ".nii" in name:
        try:
            import nibabel as nib  # type: ignore
            from PIL import Image  # type: ignore

            img = nib.load(io.BytesIO(raw))  # type: ignore
            data = img.get_fdata()  # type: ignore
            mid = data.shape[2] // 2 if data.ndim == 3 else 0
            sl = data[:, :, mid] if data.ndim == 3 else data
            sl = (sl - sl.min()) / (float(sl.max() - sl.min()) + 1e-8)
            im = Image.fromarray((sl * 255).astype("uint8")).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
            return "image/jpeg", base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            logger.warning("NIfTI conversion failed: %s", e)
            return None

    if name.endswith(".pdf") or ct == "application/pdf":
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(io.BytesIO(raw)) as pdf:  # type: ignore
                text = "\n".join((p.extract_text() or "") for p in pdf.pages[:8])
            if text.strip():
                return "text/plain", base64.b64encode(text.encode("utf-8")).decode("ascii")
        except Exception as e:
            logger.warning("PDF extraction failed: %s", e)

    return None


def maybe_text_from_bytes(raw: bytes) -> Optional[str]:
    try:
        return raw.decode("utf-8")
    except Exception:
        try:
            return raw.decode("latin-1")
        except Exception:
            return None
