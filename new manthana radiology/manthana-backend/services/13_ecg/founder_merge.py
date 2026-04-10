"""Load ecg_label_map.v1.json and merge external founder scores into legacy RHYTHM_KEYS."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ecg_rhythm import RHYTHM_KEYS

logger = logging.getLogger("manthana.ecg_founder_merge")

_DIR = Path(__file__).resolve().parent
_DEFAULT_MAP = _DIR / "ecg_label_map.v1.json"


def load_label_map_entries(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or _DEFAULT_MAP
    if not p.is_file():
        logger.warning("ECG label map missing: %s", p)
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("ECG label map invalid %s: %s", p, e)
        return []
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict) and "match" in e and "legacy_weights" in e]


def merge_founder_into_rhythm_scores(
    founder: dict[str, float],
    heuristic: dict[str, float],
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Start from heuristic scores; for each founder (key, value), if key matches an entry
    substring, raise legacy keys toward max(heuristic, value * weight).
    """
    ent = entries if entries is not None else load_label_map_entries()
    out = {k: float(heuristic.get(k, 0.0) or 0.0) for k in RHYTHM_KEYS}

    for fk, fv in founder.items():
        if not isinstance(fk, str) or not isinstance(fv, (int, float)):
            continue
        fv = float(np.clip(float(fv), 0.0, 1.0))
        fk_l = fk.lower()
        for e in ent:
            match = str(e.get("match", "")).lower()
            if not match or match not in fk_l:
                continue
            lw = e.get("legacy_weights")
            if not isinstance(lw, dict):
                continue
            for lk, w in lw.items():
                if lk not in out:
                    continue
                try:
                    wt = float(w)
                except (TypeError, ValueError):
                    continue
                contrib = float(np.clip(fv * wt, 0.0, 1.0))
                out[lk] = float(np.clip(max(out[lk], contrib), 0.0, 1.0))
    return out
