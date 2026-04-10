"""
Optional multi-class ICH subtype head (TorchScript).
Deploy: set CT_BRAIN_SUBTYPE_MODEL_PATH to a JIT bundle that outputs 5 logits
(intraparenchymal, subdural, epidural, subarachnoid, intraventricular) or softmax probs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.ct_brain.subtype")

SUBTYPE_KEYS = (
    "intraparenchymal",
    "subdural",
    "epidural",
    "subarachnoid",
    "intraventricular",
)

_subtype_model: Any = None


def _torch_device():
    import torch

    if os.getenv("CT_BRAIN_DEVICE", "").strip().lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_subtype_model() -> Any | None:
    global _subtype_model
    path = (os.environ.get("CT_BRAIN_SUBTYPE_MODEL_PATH") or "").strip()
    if not path or not os.path.isfile(path):
        return None
    if _subtype_model is not None:
        return _subtype_model
    import torch

    try:
        _subtype_model = torch.jit.load(path, map_location=_torch_device())
        _subtype_model.eval()
        logger.info("Loaded CT Brain subtype TorchScript from %s", path)
    except Exception as e:
        logger.warning("CT Brain subtype model load failed: %s", e)
        return None
    return _subtype_model


def run_subtype_classifier(tensor_5d: Any) -> dict[str, float] | None:
    """
    tensor_5d: (1,1,D,H,W) torch tensor, same preprocess as main ICH model.
    Returns dict of 0-1 scores per subtype, or None if model unavailable.
    """
    m = _load_subtype_model()
    if m is None or tensor_5d is None:
        return None
    import torch

    dev = _torch_device()
    t = tensor_5d.to(dev)
    try:
        with torch.no_grad():
            out = m(t)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.float().reshape(-1)
        n = out.numel()
        if n == 5:
            probs = torch.softmax(out, dim=-1).cpu().numpy()
            return {SUBTYPE_KEYS[i]: float(probs[i]) for i in range(5)}
        if n > 5:
            probs = torch.softmax(out[:5], dim=-1).cpu().numpy()
            return {SUBTYPE_KEYS[i]: float(probs[i]) for i in range(5)}
        if n >= 2:
            # binary head — map positive class to intraparenchymal only
            p = float(torch.softmax(out[:2], dim=-1)[1].item())
            return {
                "intraparenchymal": p,
                "subdural": 0.0,
                "epidural": 0.0,
                "subarachnoid": 0.0,
                "intraventricular": 0.0,
            }
    except Exception as e:
        logger.warning("CT Brain subtype inference failed: %s", e)
    return None


def reset_subtype_cache_for_tests() -> None:
    global _subtype_model
    _subtype_model = None
