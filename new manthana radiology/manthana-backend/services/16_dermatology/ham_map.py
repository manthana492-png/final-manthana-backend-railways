"""
Map HAM10000-style 7-class probabilities to legacy DERM_CLASSES (12).

Default HAM7 order: akiec, bcc, bkl, df, mel, nv, vasc — must match checkpoint (see config.HAM7_CLASS_ORDER).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from classifier import DERM_CLASSES

logger = logging.getLogger("manthana.dermatology.ham_map")

# Rows: HAM7 keys in config order; columns: DERM_CLASSES; row stochastic.
# akiec → SCC/BCC/melanoma-inflammatory spread; mel → melanoma; nv/bkl/df/vasc → benign mass
_HAM_TO_DERM_WEIGHTS: dict[str, dict[str, float]] = {
    "akiec": {
        "scc": 0.40,
        "bcc": 0.25,
        "melanoma": 0.15,
        "eczema_dermatitis": 0.10,
        "psoriasis": 0.05,
        "normal_benign": 0.05,
    },
    "bcc": {"bcc": 1.0},
    "bkl": {"normal_benign": 0.85, "melasma": 0.10, "psoriasis": 0.05},
    "df": {"normal_benign": 1.0},
    "mel": {"melanoma": 1.0},
    "nv": {"normal_benign": 0.72, "melanoma": 0.18, "bcc": 0.10},
    "vasc": {"normal_benign": 0.55, "urticaria": 0.30, "eczema_dermatitis": 0.15},
}


def ham7_probs_to_derm_scores(
    ham_probs: dict[str, float],
    ham7_order: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Returns (raw_ham7_scores_for_structures, full_condition_scores including meta keys).
    """
    vec = np.zeros(7, dtype=np.float64)
    for i, key in enumerate(ham7_order):
        vec[i] = float(np.clip(float(ham_probs.get(key, 0.0) or 0.0), 0.0, 1.0))

    s = vec.sum()
    if s > 1e-9:
        vec = vec / s

    M = np.zeros((7, len(DERM_CLASSES)), dtype=np.float64)
    for i, hkey in enumerate(ham7_order):
        wmap = _HAM_TO_DERM_WEIGHTS.get(hkey)
        if not wmap:
            logger.warning("Unknown HAM7 key %s — skipping mapping row", hkey)
            continue
        tw = sum(wmap.values())
        if tw <= 0:
            continue
        for j, dclass in enumerate(DERM_CLASSES):
            M[i, j] = float(wmap.get(dclass, 0.0)) / tw

    out_vec = vec @ M
    out_vec = np.clip(out_vec, 0.0, 1.0)
    tot = out_vec.sum()
    if tot > 1e-9:
        out_vec = out_vec / tot

    scores: dict[str, Any] = {
        DERM_CLASSES[j]: round(float(out_vec[j]), 4) for j in range(len(DERM_CLASSES))
    }
    top = max(scores, key=scores.get)
    conf = scores[top]
    scores["top_class"] = top
    scores["confidence"] = conf
    scores["confidence_label"] = (
        "high" if conf >= 0.70 else "medium" if conf >= 0.45 else "low"
    )
    scores["is_malignant_candidate"] = top in {"bcc", "scc", "melanoma"}

    raw7 = {k: round(float(ham_probs.get(k, 0.0) or 0.0), 4) for k in ham7_order}
    return raw7, scores


def ham_malignancy_hint(ham_probs: dict[str, float], ham7_order: tuple[str, ...]) -> dict[str, float]:
    """Raw mass on cancer-precursor classes for critical_flags boost."""
    mel = float(ham_probs.get("mel", 0.0) or 0.0)
    bcc = float(ham_probs.get("bcc", 0.0) or 0.0)
    akiec = float(ham_probs.get("akiec", 0.0) or 0.0)
    return {
        "ham_mel": mel,
        "ham_bcc": bcc,
        "ham_akiec": akiec,
        "ham_combined_malignancy": float(mel + bcc + 0.5 * akiec),
    }
