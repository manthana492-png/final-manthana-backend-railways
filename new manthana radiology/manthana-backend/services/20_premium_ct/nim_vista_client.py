"""NVIDIA NIM VISTA-3D hosted inference (Premium 3D CT).

See https://docs.nvidia.com/nim/medical/vista3d/latest/api-reference.html
The inference API expects ``image`` as an HTTPS URL to NIfTI/NRRD. For production, sync the
exported volume to a URL reachable by NVIDIA (e.g. object storage + public or signed URL), or set
``NIM_VISTA_IMAGE_URL`` for integration tests.

Environment:
  NVIDIA_NIM_API_KEY — required for hosted NIM
  NVIDIA_NIM_VISTA_INFER_URL — POST endpoint (default: health.api.nvidia.com medical imaging path)
  NIM_VISTA_IMAGE_URL — optional full URL override for the volume (testing)
  NIM_VISTA_PUBLIC_BASE_URL + NIM_VISTA_PUBLIC_SYNC_DIR — copy ``{job_id}_volume.nii.gz`` to sync dir
    and set image URL to ``{base}/{job_id}_volume.nii.gz``
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import requests

from segmentation_export import export_ct_volume_nifti_gz
from vista3d_labels import LABEL_DICT

logger = logging.getLogger("manthana.premium_ct.nim_vista")

_DEFAULT_INFER = (
    "https://health.api.nvidia.com/v1/medicalimaging/nvidia/vista-3d/infer"
)


def _parse_segmentation_bytes(raw: bytes) -> np.ndarray:
    """NIM returns segmentation as NIfTI bytes (application/octet-stream) per API docs."""
    try:
        img = nib.load(BytesIO(raw))
        return np.asarray(img.dataobj)
    except Exception:
        pass
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise ValueError("NIM VISTA response: not NIfTI bytes or JSON") from None
    b64 = data.get("segmentation") or data.get("mask") or data.get("output")
    if isinstance(b64, str) and b64.strip():
        import base64

        decoded = base64.b64decode(b64)
        img = nib.load(BytesIO(decoded))
        return np.asarray(img.dataobj)
    raise ValueError("NIM VISTA JSON response did not contain segmentation payload")


def _build_class_scores_from_mask(seg_np: np.ndarray) -> dict[str, float]:
    """Approximate class scores from discrete label map (NIM returns mask without softmax)."""
    class_scores: dict[str, float] = {}
    for class_idx in np.unique(seg_np):
        if int(class_idx) == 0:
            continue
        label = LABEL_DICT.get(int(class_idx), f"class_{int(class_idx)}")
        class_scores[label] = 1.0
    return class_scores


def _volumes_ml_from_mask(
    seg_np: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, float]:
    from segmentation_export import volume_ml_for_class

    out: dict[str, float] = {}
    for class_idx in np.unique(seg_np):
        if int(class_idx) == 0:
            continue
        label = LABEL_DICT.get(int(class_idx), f"class_{int(class_idx)}")
        out[label] = volume_ml_for_class(seg_np, int(class_idx), spacing_xyz)
    return out


def resolve_nifti_public_url(*, job_id: str, local_path: Path) -> str:
    """Determine HTTPS URL for the NIfTI file passed to NIM ``image`` field."""
    override = (os.environ.get("NIM_VISTA_IMAGE_URL") or "").strip()
    if override:
        return override

    base = (os.environ.get("NIM_VISTA_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    sync_dir = (os.environ.get("NIM_VISTA_PUBLIC_SYNC_DIR") or "").strip()
    if base and sync_dir:
        dest = Path(sync_dir) / f"{job_id}_volume.nii.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"{base}/{job_id}_volume.nii.gz"

    raise ValueError(
        "NIM VISTA requires a public URL for the NIfTI volume. Set NIM_VISTA_IMAGE_URL, or "
        "NIM_VISTA_PUBLIC_BASE_URL + NIM_VISTA_PUBLIC_SYNC_DIR (see services/20_premium_ct README). "
        "Reference: https://docs.nvidia.com/nim/medical/vista3d/latest/api-reference.html"
    )


def run_vista3d_segmentation_nim(
    *,
    volume: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    region_hint: str | None,
    job_id: str,
) -> dict[str, Any]:
    """Run VISTA-3D via NVIDIA NIM; return the same shape as ``run_vista3d_segmentation``."""
    api_key = (os.environ.get("NVIDIA_NIM_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("NVIDIA_NIM_API_KEY is not set")

    infer_url = (os.environ.get("NVIDIA_NIM_VISTA_INFER_URL") or _DEFAULT_INFER).strip()
    t0 = time.time()

    out_dir = Path(os.environ.get("NIM_VISTA_TMP_DIR", "/tmp/manthana_nim_vista"))
    local_nii = out_dir / f"{job_id}_volume.nii.gz"
    export_ct_volume_nifti_gz(volume, local_nii, spacing_xyz)
    image_url = resolve_nifti_public_url(job_id=job_id, local_path=local_nii)

    payload: dict[str, Any] = {"image": image_url}
    if region_hint and region_hint not in ("", "full_body"):
        payload["prompts"] = {"classes": [region_hint]}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/octet-stream, application/json",
    }

    logger.info(
        "nim_vista_infer model_provider=nim nim_infer_url=%s job_id=%s elapsed_setup_sec=%.2f",
        infer_url,
        job_id,
        time.time() - t0,
    )

    resp = requests.post(infer_url, json=payload, headers=headers, timeout=600)
    if not resp.ok:
        logger.error(
            "nim_vista_infer_failed status=%s body=%s",
            resp.status_code,
            resp.text[:2000],
        )
        resp.raise_for_status()

    seg_np = _parse_segmentation_bytes(resp.content)
    seg_u8 = np.asarray(seg_np).astype(np.uint8)

    class_scores = _build_class_scores_from_mask(seg_u8)
    volumes_ml = _volumes_ml_from_mask(seg_u8, spacing_xyz)

    return {
        "segmentation_mask": seg_u8,
        "class_scores": class_scores,
        "volumes_ml": volumes_ml,
        "classes_detected": len(class_scores),
        "region_analyzed": region_hint or "full_body",
        "nim_infer_url": infer_url,
        "nim_image_url": image_url,
        "nim_latency_sec": round(time.time() - t0, 2),
    }
