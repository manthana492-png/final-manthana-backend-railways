"""
Manthana — TorchXRayVision shared utilities (single preprocessing + model singletons).

Canonical contract: 18 logits per checkpoint indexed by xrv.datasets.default_pathologies;
model.pathologies[i] empty means untrained slot — exclude from filtered scores.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("manthana.txrv_utils")

# ── Weight ids (RSNA must never appear in ensemble — regression-tested) ─────
PRIMARY_WEIGHTS = "densenet121-res224-all"
SECONDARY_WEIGHTS = "densenet121-res224-chex"
FALLBACK_SECONDARY = "densenet121-res224-mimic_nb"

_models: dict[str, Any] = {}
_secondary_weights_used: str | None = None
_load_stats: dict[str, dict[str, float | bool]] = {}
_model_residency: dict[str, str] = {}
_last_used_epoch_s: dict[str, float] = {}
_active_inferences: int = 0
_lock = threading.RLock()
_reaper_started = False

IDLE_UNLOAD_SEC = float(os.getenv("XRAY_MODEL_IDLE_UNLOAD_SEC", "0") or "0")
UNLOAD_MODE = os.getenv("XRAY_UNLOAD_MODE", "gpu_only").strip().lower()
IDLE_CHECK_SEC = float(os.getenv("XRAY_MODEL_IDLE_CHECK_SEC", "15") or "15")
if UNLOAD_MODE not in {"gpu_only", "full"}:
    logger.warning("Invalid XRAY_UNLOAD_MODE=%s; defaulting to gpu_only", UNLOAD_MODE)
    UNLOAD_MODE = "gpu_only"


def _touch(weights: str) -> None:
    _last_used_epoch_s[weights] = time.time()


def _detect_residency(model: Any) -> str:
    try:
        p = next(model.parameters())
        return "gpu" if p.is_cuda else "cpu"
    except Exception:
        return "cpu"


class _InferenceGuard:
    def __enter__(self):
        global _active_inferences
        with _lock:
            _active_inferences += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        global _active_inferences
        with _lock:
            _active_inferences = max(0, _active_inferences - 1)


def _ensure_reaper_started() -> None:
    global _reaper_started
    if _reaper_started or IDLE_UNLOAD_SEC <= 0:
        return
    with _lock:
        if _reaper_started:
            return

        def _loop() -> None:
            while True:
                try:
                    time.sleep(max(1.0, IDLE_CHECK_SEC))
                    run_idle_unload_check()
                except Exception as e:
                    logger.warning("TXRV idle reaper loop error: %s", e)

        t = threading.Thread(target=_loop, name="txrv-idle-reaper", daemon=True)
        t.start()
        _reaper_started = True
        logger.info(
            "TXRV idle unload enabled (sec=%s, mode=%s, check=%ss)",
            IDLE_UNLOAD_SEC,
            UNLOAD_MODE,
            IDLE_CHECK_SEC,
        )


def run_idle_unload_check() -> dict[str, Any]:
    """
    Unload idle TXRV models safely. Intended for background reaper and tests.
    """
    if IDLE_UNLOAD_SEC <= 0:
        return {"enabled": False, "unloaded": []}
    now = time.time()
    unloaded: list[str] = []
    with _lock:
        if _active_inferences > 0:
            return {"enabled": True, "unloaded": [], "skipped_active_inferences": True}
        for weights in list(_models.keys()):
            last = _last_used_epoch_s.get(weights, now)
            idle_for = now - last
            if idle_for < IDLE_UNLOAD_SEC:
                continue
            model = _models.get(weights)
            if model is None:
                continue
            if UNLOAD_MODE == "full":
                _models.pop(weights, None)
                _model_residency.pop(weights, None)
                _last_used_epoch_s.pop(weights, None)
                unloaded.append(weights)
                continue
            # gpu_only: move to CPU but keep in RAM.
            try:
                if _model_residency.get(weights) == "gpu":
                    model = model.cpu()
                    _models[weights] = model
                    _model_residency[weights] = "cpu"
                    unloaded.append(weights)
            except Exception as e:
                logger.warning("Failed to move %s to CPU during idle unload: %s", weights, e)
        if torch.cuda.is_available() and unloaded:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return {"enabled": True, "mode": UNLOAD_MODE, "unloaded": unloaded}

# TXRV label → Manthana canonical snake_case (correlation_engine / report assembly)
TXRV_LABEL_MAP: dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "Infiltration": "infiltration",
    "Pneumothorax": "pneumothorax",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia",
    "Pleural_Thickening": "pleural_thickening",
    "Cardiomegaly": "cardiomegaly",
    "Nodule": "nodule",
    "Mass": "mass",
    "Hernia": "hernia",
    "Lung Lesion": "lung_lesion",
    "Fracture": "fracture",
    "Lung Opacity": "lung_opacity",
    "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum",
    "No Finding": "no_finding",
}


def txrv_tensor_from_filepath(filepath: str) -> torch.Tensor:
    """
    Canonical TXRV input: [1, 1, 224, 224] float32 on CPU (move to CUDA in caller).
    """
    import skimage.io
    import torchxrayvision as xrv
    from torchvision import transforms

    img = skimage.io.imread(filepath)
    img = xrv.datasets.normalize(img, 255)
    if len(img.shape) > 2:
        img = img.mean(axis=2)
    if len(img.shape) < 2:
        raise ValueError(f"Unexpected image dimensions: {img.shape} for file {filepath}")
    img = img[None, :, :]
    transform = transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )
    img = transform(img)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).unsqueeze(0)
    if os.getenv("TXRV_DEBUG_PREPROCESS", "").lower() in ("1", "true", "yes"):
        logger.debug(
            "TXRV input: dtype=%s shape=%s min=%.2f max=%.2f",
            t.dtype,
            tuple(t.shape),
            float(t.min()),
            float(t.max()),
        )
    return t


def _load_txrv(weights: str):
    import torchxrayvision as xrv

    global _models
    with _lock:
        _ensure_reaper_started()
        if weights not in _models:
            logger.info("Loading TorchXRayVision weights=%s", weights)
            t0 = time.perf_counter()
            m = xrv.models.DenseNet(weights=weights)
            m.eval()
            if torch.cuda.is_available():
                m = m.cuda()
            _models[weights] = m
            _model_residency[weights] = _detect_residency(m)
            _touch(weights)
            _load_stats[weights] = {
                "loaded": True,
                "cold_load_seconds": round(time.perf_counter() - t0, 3),
                "loaded_at_epoch_s": round(time.time(), 3),
                "residency": _model_residency[weights],
            }
            return m

        m = _models[weights]
        # gpu_only unload may keep model in CPU RAM; promote back to GPU on demand.
        if torch.cuda.is_available() and _model_residency.get(weights) == "cpu":
            t0 = time.perf_counter()
            m = m.cuda()
            _models[weights] = m
            _model_residency[weights] = "gpu"
            st = _load_stats.setdefault(weights, {})
            st["warm_promote_seconds"] = round(time.perf_counter() - t0, 3)
            st["residency"] = "gpu"
        _touch(weights)
        return m


def get_txrv_all():
    return _load_txrv(PRIMARY_WEIGHTS)


def is_primary_loaded() -> bool:
    """True once densenet121-res224-all is resident in the process."""
    return PRIMARY_WEIGHTS in _models


def get_secondary_model() -> tuple[Any, str]:
    """Returns (model, weights_id) — chex or mimic_nb fallback."""
    global _secondary_weights_used
    try:
        m = _load_txrv(SECONDARY_WEIGHTS)
        _secondary_weights_used = SECONDARY_WEIGHTS
        return m, SECONDARY_WEIGHTS
    except Exception as e:
        logger.warning(
            "Secondary weights %s failed (%s); using fallback %s",
            SECONDARY_WEIGHTS,
            e,
            FALLBACK_SECONDARY,
        )
        m = _load_txrv(FALLBACK_SECONDARY)
        _secondary_weights_used = FALLBACK_SECONDARY
        return m, FALLBACK_SECONDARY


def normalize_txrv_scores(raw: dict[str, float]) -> dict[str, float]:
    """Map TXRV pathology names to Manthana canonical keys."""
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k is None or k == "":
            continue
        canon = TXRV_LABEL_MAP.get(k)
        if canon is None:
            canon = k.lower().replace(" ", "_").replace("-", "_")
        out[canon] = float(v)
    return out


def run_txrv(filepath: str, weights: str) -> dict[str, float]:
    """
    Run a single TXRV checkpoint; return canonical snake_case pathology scores.
    """
    import torchxrayvision as xrv

    model = _load_txrv(weights)
    with _InferenceGuard():
        tensor = txrv_tensor_from_filepath(filepath)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            logits = model(tensor)
    _touch(weights)

    assert logits.shape[-1] == len(xrv.datasets.default_pathologies), (
        f"TXRV output length {logits.shape[-1]} != "
        f"default_pathologies length {len(xrv.datasets.default_pathologies)}. "
        "Check torchxrayvision version pin."
    )

    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    labs = model.pathologies
    filtered: dict[str, float] = {}
    for i, name in enumerate(xrv.datasets.default_pathologies):
        if i >= len(labs):
            break
        lab = labs[i]
        if lab:
            filtered[name] = float(probs[i])
    return normalize_txrv_scores(filtered)


def ensemble_txrv(filepath: str) -> tuple[dict[str, float], float, list[str]]:
    """
    Dual TXRV ensemble (all + chex/mimic_nb). Returns:
      (canonical pathology_scores, agreement in [0,1], models_used internal strings)
    """
    scores_primary = run_txrv(filepath, PRIMARY_WEIGHTS)
    _, sec_w = get_secondary_model()
    scores_secondary = run_txrv(filepath, sec_w)

    shared = (set(scores_primary.keys()) & set(scores_secondary.keys())) - {"no_finding"}
    averaged: dict[str, float] = {}
    for k in shared:
        averaged[k] = (scores_primary[k] + scores_secondary[k]) / 2.0

    primary_only = set(scores_primary.keys()) - set(scores_secondary.keys()) - {"no_finding"}
    for k in primary_only:
        averaged[k] = scores_primary[k]

    if "no_finding" in scores_primary:
        averaged["no_finding"] = scores_primary["no_finding"]
    elif "no_finding" in scores_secondary:
        averaged["no_finding"] = scores_secondary["no_finding"]

    diffs = [abs(scores_primary[k] - scores_secondary[k]) for k in shared]
    agreement = float(1.0 - (sum(diffs) / len(diffs))) if diffs else 0.5

    secondary_tag = (
        "TorchXRayVision-DenseNet121-chex"
        if sec_w == SECONDARY_WEIGHTS
        else "TorchXRayVision-DenseNet121-mimic_nb"
    )  # sec_w from get_secondary_model()
    models_used = [
        "TorchXRayVision-DenseNet121-all",
        secondary_tag,
    ]
    return averaged, agreement, models_used


def triage_forward_probs(path: str) -> tuple[np.ndarray, list[str]]:
    """For gateway triage: all-18 sigmoid probs + default_pathology names."""
    import torchxrayvision as xrv

    model = get_txrv_all()
    with _InferenceGuard():
        tensor = txrv_tensor_from_filepath(path)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            logits = model(tensor)
    _touch(PRIMARY_WEIGHTS)
    assert logits.shape[-1] == len(xrv.datasets.default_pathologies)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    names = list(xrv.datasets.default_pathologies)
    return probs, names


def get_txrv_runtime_stats() -> dict[str, Any]:
    """
    Runtime observability for cold/warm behavior and loaded checkpoints.
    """
    return {
        "loaded_weights": sorted(list(_models.keys())),
        "primary_loaded": PRIMARY_WEIGHTS in _models,
        "secondary_loaded": SECONDARY_WEIGHTS in _models or FALLBACK_SECONDARY in _models,
        "secondary_weights_used": _secondary_weights_used,
        "load_stats": dict(_load_stats),
        "residency": dict(_model_residency),
        "idle_policy": {
            "enabled": IDLE_UNLOAD_SEC > 0,
            "idle_unload_sec": IDLE_UNLOAD_SEC,
            "mode": UNLOAD_MODE,
            "check_sec": IDLE_CHECK_SEC,
        },
        "active_inferences": _active_inferences,
    }
