"""
Medical image conversion and de-identification for Manthana Labs gateway.
Converts DICOM / ambiguous binary to PNG before sending to vision LLMs.
Extracts text from PDF before sending to text LLMs.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_PHI_TAGS_TO_BLANK = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "RequestingPhysician",
    "OperatorsName",
    "PhysiciansOfRecord",
    "StudyID",
    "AccessionNumber",
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "RequestedProcedureDescription",
    "CountryOfResidence",
    "RegionOfResidence",
    "PatientInsurancePlanCodeSequence",
]


def _deidentify_dicom(ds) -> None:
    from pydicom.uid import generate_uid

    for tag_name in _PHI_TAGS_TO_BLANK:
        if hasattr(ds, tag_name):
            try:
                setattr(ds, tag_name, "")
            except Exception:
                try:
                    delattr(ds, tag_name)
                except Exception:
                    pass
    ds.remove_private_tags()
    for uid_attr in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        if hasattr(ds, uid_attr):
            try:
                setattr(ds, uid_attr, generate_uid())
            except Exception:
                pass


def _dicom_to_png_b64(raw_bytes: bytes) -> str:
    import numpy as np
    import pydicom
    from PIL import Image

    ds = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
    _deidentify_dicom(ds)

    pixel_array = ds.pixel_array.astype(np.float32)

    if pixel_array.ndim == 3:
        last = pixel_array.shape[2]
        if last in (1, 2, 3, 4) and last <= 4:
            pass  # (H, W, C) colour / multi-channel
        elif pixel_array.shape[0] > 3:
            mid = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid]

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        center = (
            float(ds.WindowCenter)
            if not hasattr(ds.WindowCenter, "__iter__")
            else float(list(ds.WindowCenter)[0])
        )
        width = (
            float(ds.WindowWidth)
            if not hasattr(ds.WindowWidth, "__iter__")
            else float(list(ds.WindowWidth)[0])
        )
        low = center - width / 2
        high = center + width / 2
        pixel_array = np.clip(pixel_array, low, high)

    pmin, pmax = float(pixel_array.min()), float(pixel_array.max())
    if pmax > pmin:
        pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = 255 - pixel_array

    img = Image.fromarray(pixel_array)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if img.mode == "L":
        img = img.convert("RGB")

    max_px = 2048
    if max(img.size) > max_px:
        ratio = max_px / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def convert_image_for_llm(image_b64: str, image_mime: str) -> Tuple[str, str]:
    if not image_b64:
        return image_b64, image_mime

    mime = (image_mime or "").strip().lower()
    if mime in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"):
        return image_b64, image_mime

    if mime == "application/pdf" or mime.endswith("/pdf"):
        raise ValueError(
            "PDF files must be processed with extract_pdf_text(), not convert_image_for_llm()."
        )

    if (
        "dicom" in mime
        or image_mime in ("application/octet-stream", "", None)
        or not image_mime
    ):
        try:
            raw_bytes = base64.b64decode(image_b64)
            converted_b64 = _dicom_to_png_b64(raw_bytes)
            logger.info("DICOM converted to PNG successfully")
            return converted_b64, "image/png"
        except Exception as e:
            raise ValueError(f"DICOM to PNG conversion failed: {e}") from e

    logger.warning("Unknown image mime type %r — passing through unconverted.", image_mime)
    return image_b64, image_mime


def extract_pdf_text(image_b64: str, max_chars: int = 8000) -> str:
    try:
        import pymupdf

        raw = base64.b64decode(image_b64)
        doc = pymupdf.open(stream=raw, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        full_text = "\n".join(pages_text).strip()
        if not full_text:
            logger.warning("PDF had no extractable text layer.")
            return (
                "[Document appears to be a scanned image. Text extraction not available. "
                "Please upload a text-based PDF or a photo of the document.]"
            )
        return full_text[:max_chars]
    except Exception as e:
        raise ValueError(f"PDF text extraction failed: {e}") from e


def is_pdf(image_mime: str) -> bool:
    m = image_mime or ""
    return m == "application/pdf" or m.endswith("/pdf")


def is_dicom_or_binary(image_mime: Optional[str]) -> bool:
    m = (image_mime or "").lower()
    return "dicom" in m or image_mime in ("application/octet-stream", "", None) or not image_mime


def merge_pdf_into_patient_context_json(existing_json: Optional[str], pdf_text: str, cap: int = 12000) -> str:
    try:
        import json as _json

        ctx = _json.loads(existing_json or "{}")
    except Exception:
        ctx = {}
    ctx["extracted_document_text"] = pdf_text
    raw = _json.dumps(ctx, ensure_ascii=False)
    return raw[:cap]
