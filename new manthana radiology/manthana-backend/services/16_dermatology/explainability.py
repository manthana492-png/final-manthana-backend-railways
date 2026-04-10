"""
Optional attention-style map for torchvision EfficientNet (B4 / V2-M).

Uses Grad-CAM-style weights from gradients w.r.t. the last feature map (input to avgpool).
Returns None on any failure — never raises to callers.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("manthana.dermatology.explainability")


def _efficientnet_gradcam_base64(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    device: torch.device,
    pil_size: tuple[int, int],
) -> str | None:
    """input_tensor (1,3,H,W) normalized; pil_size (W,H) for resize overlay."""
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_m: Any, _inp: Any, out: torch.Tensor) -> None:
        activations.clear()
        activations.append(out)

    def full_bwd_hook(_m: Any, gi: tuple[torch.Tensor | None, ...], _go: Any) -> None:
        # Gradient w.r.t. feature map (avgpool input), not pooled output
        if gi and gi[0] is not None:
            gradients.clear()
            gradients.append(gi[0])

    if not hasattr(model, "avgpool") or not hasattr(model, "features"):
        return None

    h1 = model.features.register_forward_hook(fwd_hook)
    h2 = model.avgpool.register_full_backward_hook(full_bwd_hook)
    try:
        x = input_tensor.clone().detach().requires_grad_(True)
        out = model(x)
        if target_class_idx < 0 or target_class_idx >= out.shape[1]:
            return None
        score = out[0, target_class_idx]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        if not activations or not gradients:
            return None
        act = activations[0]
        grad = gradients[0]
        if act is None or grad is None:
            return None
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = (w * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.size == 0:
            return None
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
        heat = Image.fromarray(cam_u8, mode="L").resize(pil_size, Image.Resampling.BILINEAR)
        # red overlay
        heat_rgb = Image.merge(
            "RGB",
            (
                heat,
                Image.new("L", pil_size, 0),
                Image.new("L", pil_size, 0),
            ),
        )
        base = Image.new("RGB", pil_size, (0, 0, 0))
        blended = Image.blend(base, heat_rgb, alpha=0.45)
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.debug("Grad-CAM skipped: %s", e)
        return None
    finally:
        h1.remove()
        h2.remove()


def try_explainability_png(
    model: torch.nn.Module,
    pil_image: Image.Image,
    transform: Any,
    derm_class_order: list[str],
    top_class: str,
    device: str,
) -> str | None:
    """Returns base64 PNG or None."""
    try:
        if top_class not in derm_class_order:
            return None
        idx = derm_class_order.index(top_class)
        t = transform(pil_image).unsqueeze(0).to(device)
        dev = torch.device(device)
        w, h = pil_image.size
        return _efficientnet_gradcam_base64(model, t, idx, dev, (w, h))
    except Exception as e:
        logger.debug("explainability: %s", e)
        return None
