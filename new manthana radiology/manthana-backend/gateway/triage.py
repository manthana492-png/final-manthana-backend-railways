# pyright: reportMissingImports=false
"""Manthana — Triage Layer (lightweight screening before deep analysis)."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np

# Gateway lives in manthana-backend/gateway/; shared is ../shared (Docker: /app/shared)
_GATEWAY_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.normpath(os.path.join(_GATEWAY_DIR, ".."))
sys.path.insert(0, os.path.join(_BACKEND_ROOT, "shared"))
if os.path.isdir("/app/shared"):
    sys.path.insert(0, "/app/shared")

# Lazy-import txrv_utils inside _triage_xray only — avoids loading PyTorch/torchxrayvision
# at gateway startup when XRAY_TRIAGE_POLICY=always_deep (default) and CPU-only Railway.

logger = logging.getLogger("manthana.gateway.triage")

ABNORMALITY_THRESHOLD = float(os.getenv("TRIAGE_ABNORMALITY_THRESHOLD", "0.3"))
XRAY_TRIAGE_POLICY = os.getenv("XRAY_TRIAGE_POLICY", "always_deep").strip().lower()


def _load_xray_gray(path: str) -> np.ndarray | None:
    try:
        from PIL import Image

        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception as e:
        logger.warning("Triage X-ray load failed: %s", e)
        return None


def _triage_xray(path: str) -> dict[str, Any]:
    """
    DenseNet121 torchxrayvision — fast pathology screening (shared singleton + preprocess).
    """
    from txrv_utils import get_txrv_runtime_stats, triage_forward_probs  # noqa: E402

    t0 = time.perf_counter()
    try:
        probs, names = triage_forward_probs(path)
        max_p = float(np.max(probs))
        scores = {names[i]: float(probs[i]) for i in range(len(names))}
        needs_deep = max_p >= ABNORMALITY_THRESHOLD
        triage_ms = int((time.perf_counter() - t0) * 1000)
        findings = {
            "label": "triage",
            "severity": "warning" if needs_deep else "clear",
            "confidence": max_p * 100,
            "description": f"Max screening probability {max_p:.2f} on {names[int(np.argmax(probs))]}",
        }
        return {
            "needs_deep": needs_deep,
            "findings": [findings],
            "triage_scores": scores,
            "triage_time_ms": triage_ms,
            "models_used": ["TorchXRayVision-DenseNet121-triage"],
            "runtime": get_txrv_runtime_stats(),
        }
    except Exception as e:
        logger.warning("X-ray triage fallback (heuristic): %s", e)
        return _triage_xray_heuristic(path)


def _triage_xray_heuristic(path: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    g = _load_xray_gray(path)
    if g is None:
        return {
            "needs_deep": True,
            "findings": [],
            "triage_scores": {},
            "triage_time_ms": int((time.perf_counter() - t0) * 1000),
            "models_used": ["triage-heuristic"],
        }
    std = float(np.std(g))
    # High texture / contrast → more likely abnormal
    score = min(1.0, std * 4.0)
    needs_deep = score >= ABNORMALITY_THRESHOLD
    triage_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "needs_deep": needs_deep,
        "findings": [
            {
                "label": "triage",
                "severity": "warning" if needs_deep else "clear",
                "confidence": score * 100,
                "description": f"Heuristic texture score {score:.2f}",
            }
        ],
        "triage_scores": {"heuristic_abnormality": score},
        "triage_time_ms": triage_ms,
        "models_used": ["triage-heuristic"],
        "runtime": {"mode": "heuristic"},
    }


def _triage_default(path: str, modality: str) -> dict[str, Any]:
    """Non-CXR modalities: always route to deep (no lightweight model in gateway)."""
    t0 = time.perf_counter()
    _ = path
    return {
        "needs_deep": True,
        "findings": [],
        "triage_scores": {},
        "triage_time_ms": int((time.perf_counter() - t0) * 1000),
        "models_used": [f"triage-pass-through-{modality}"],
    }


def run_triage(saved_path: str, modality: str) -> dict[str, Any]:
    """
    Returns:
        needs_deep, findings (list of dicts for API), triage_scores, triage_time_ms, models_used
    """
    m = (modality or "").lower().strip()
    if m in ("xray", "body_xray", "chest", "cxr"):
        if XRAY_TRIAGE_POLICY == "always_deep":
            return {
                "needs_deep": True,
                "findings": [],
                "triage_scores": {},
                "triage_time_ms": 0,
                "models_used": ["triage-policy-always-deep"],
                "runtime": {"policy": XRAY_TRIAGE_POLICY},
            }
        return _triage_xray(saved_path)
    if m in ("ecg",):
        return _triage_default(saved_path, m)
    if m in ("oral_cancer", "oral"):
        return _triage_default(saved_path, m)
    return _triage_default(saved_path, m)
