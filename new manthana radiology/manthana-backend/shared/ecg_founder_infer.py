"""
ECGFounder / external 12-lead classifier hook (Phase 1).

- If ECG_FOUNDER_CHECKPOINT points to a .json file, loads dict[str, float] (dev / contract tests).
- If it points to a .pt / .pth file, returns None until a real forward pass is implemented.
- Otherwise searches MODEL_DIR for ecg/founder/model.pt or ecg_founder.pt.

Replace the PyTorch branch with the official ECGFounder forward when weights and code are vendored.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.ecg_founder_infer")


def _candidate_paths(model_dir: str) -> list[str]:
    env = os.getenv("ECG_FOUNDER_CHECKPOINT", "").strip()
    if env:
        return [env]
    return [
        os.path.join(model_dir, "ecg", "founder", "model.pt"),
        os.path.join(model_dir, "ecg", "founder", "weights.pt"),
        os.path.join(model_dir, "ecg_founder.pt"),
        os.path.join(model_dir, "ecg_founder_scores.json"),
    ]


def founder_checkpoint_path(model_dir: str) -> str | None:
    for p in _candidate_paths(model_dir):
        if p and os.path.isfile(p):
            return p
    return None


def is_founder_checkpoint_present(model_dir: str) -> bool:
    return founder_checkpoint_path(model_dir) is not None


def predict_12lead(
    signal: np.ndarray,
    sample_rate: float,
    model_dir: str,
) -> dict[str, float] | None:
    """
    Returns founder-style label -> probability, or None if unavailable / stub.
    """
    path = founder_checkpoint_path(model_dir)
    if not path:
        return None

    if path.lower().endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("ECG founder JSON checkpoint unreadable %s: %s", path, e)
            return None
        if not isinstance(raw, dict):
            return None
        out: dict[str, float] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                out[k] = float(np.clip(float(v), 0.0, 1.0))
        return out if out else None

    # Real weights present but inference not wired yet
    logger.info(
        "ECG founder weights at %s — PyTorch forward not integrated; returning None",
        path,
    )
    return None
