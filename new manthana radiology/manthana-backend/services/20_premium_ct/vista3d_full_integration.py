"""Full VISTA-3D 127-class segmentation integration for premium CT."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vista3d_labels import LABEL_DICT

logger = logging.getLogger("manthana.premium_ct.vista3d")


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    suf = model_path.suffix.lower()
    if suf == ".safetensors" or model_path.name.endswith(".safetensors"):
        from safetensors.torch import load_file

        # Load on CPU first; ``model.to(device)`` places weights on GPU.
        raw = load_file(str(model_path))
        if not isinstance(raw, dict):
            raise RuntimeError("safetensors file did not yield a state dict.")
        return raw  # type: ignore[return-value]

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        inner = (
            checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint.get("network")
        )
        if isinstance(inner, dict):
            return inner  # type: ignore[return-value]
        return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Unsupported VISTA checkpoint format: expected dict or safetensors.")


def _load_vista_model(model_path: Path, device: str) -> torch.nn.Module:
    from monai.networks.nets import Vista3D

    model = Vista3D(in_channels=1, out_channels=128)
    state_dict = _load_state_dict(model_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _to_model_tensor(volume: np.ndarray) -> torch.Tensor:
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim == 2:
        raise ValueError("VISTA-3D requires volumetric CT. Received 2D slice.")
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape: {vol.shape}")
    lo = np.percentile(vol, 1)
    hi = np.percentile(vol, 99)
    clipped = np.clip(vol, lo, hi)
    norm = (clipped - lo) / max(hi - lo, 1e-6)
    return torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)


def run_vista3d_segmentation(
    *,
    volume: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    model_path: str,
    device: str = "cuda",
    region_hint: str | None = None,
) -> dict[str, Any]:
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"VISTA-3D checkpoint missing at: {model_file}")

    model = _load_vista_model(model_file, device)
    x = _to_model_tensor(volume).to(device)
    with torch.no_grad():
        logits = model(x)  # (1, 128, H, W, D)
        probs = torch.softmax(logits, dim=1)
        seg = torch.argmax(logits, dim=1)
    seg_np = seg[0].detach().cpu().numpy().astype(np.uint8)

    class_scores: dict[int, float] = {}
    for class_idx in np.unique(seg_np):
        if int(class_idx) == 0:
            continue
        mask = seg == int(class_idx)
        if int(mask.sum().item()) == 0:
            continue
        conf = float(probs[0, int(class_idx)][mask[0]].mean().item())
        class_scores[int(class_idx)] = round(conf, 4)

    voxel_volume_ml = (spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]) / 1000.0
    volumes_ml: dict[str, float] = {}
    for class_idx, conf in sorted(class_scores.items(), key=lambda kv: kv[1], reverse=True):
        voxel_count = int((seg_np == class_idx).sum())
        if voxel_count <= 0:
            continue
        label = LABEL_DICT.get(class_idx, f"class_{class_idx}")
        volumes_ml[label] = round(voxel_count * voxel_volume_ml, 3)

    return {
        "segmentation_mask": seg_np,
        "class_scores": {
            LABEL_DICT.get(idx, f"class_{idx}"): score for idx, score in class_scores.items()
        },
        "volumes_ml": volumes_ml,
        "classes_detected": len(class_scores),
        "region_analyzed": region_hint or "full_body",
    }


def run_vista3d_segmentation_auto(
    *,
    volume: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    model_path: str,
    device: str = "cuda",
    region_hint: str | None = None,
    job_id: str = "job",
) -> dict[str, Any]:
    """
    Primary path: NVIDIA NIM (``VISTA_BACKEND=nim``, default). Rollback: ``VISTA_BACKEND=local``.
    On NIM errors (missing URL, API failure), falls back to local MONAI checkpoint.
    """
    backend = (os.getenv("VISTA_BACKEND") or "nim").strip().lower()
    if backend == "local":
        return run_vista3d_segmentation(
            volume=volume,
            spacing_xyz=spacing_xyz,
            model_path=model_path,
            device=device,
            region_hint=region_hint,
        )
    try:
        from nim_vista_client import run_vista3d_segmentation_nim

        return run_vista3d_segmentation_nim(
            volume=volume,
            spacing_xyz=spacing_xyz,
            region_hint=region_hint,
            job_id=job_id,
        )
    except Exception as exc:
        logger.warning(
            "VISTA NIM failed (%s); falling back to local MONAI checkpoint",
            exc,
        )
        return run_vista3d_segmentation(
            volume=volume,
            spacing_xyz=spacing_xyz,
            model_path=model_path,
            device=device,
            region_hint=region_hint,
        )

