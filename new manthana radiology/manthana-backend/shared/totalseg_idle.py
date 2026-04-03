"""
TotalSegmentator GPU idle hygiene — optional `torch.cuda.empty_cache()` after idle.

TotalSeg is driven via third-party APIs; we do not own model handles for full unload.
This complements X-ray TXRV idle policy with a measurable, low-risk VRAM pressure relief.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("manthana.totalseg_idle")

IDLE_EMPTY_CACHE_SEC = float(os.getenv("CT_TOTALSEG_IDLE_EMPTY_CACHE_SEC", "0") or "0")
IDLE_CHECK_SEC = float(os.getenv("CT_TOTALSEG_IDLE_CHECK_SEC", "30") or "30")

_last_activity_monotonic: float = 0.0
_lock = threading.Lock()
_reaper_started = False


def touch_totalseg_gpu_activity() -> None:
    """Mark GPU activity (call after TotalSeg / CT inference completes)."""
    global _last_activity_monotonic
    with _lock:
        _last_activity_monotonic = time.monotonic()


def run_totalseg_idle_empty_cache_check() -> dict[str, Any]:
    """
    If idle longer than IDLE_EMPTY_CACHE_SEC, run torch.cuda.empty_cache().
    Skips when CT_TOTALSEG_IDLE_EMPTY_CACHE_SEC <= 0.
    """
    if IDLE_EMPTY_CACHE_SEC <= 0:
        return {"enabled": False, "emptied": False}

    now = time.monotonic()
    with _lock:
        last = _last_activity_monotonic
        if last <= 0:
            return {"enabled": True, "emptied": False, "reason": "never_touched"}
        idle_for = now - last
        if idle_for < IDLE_EMPTY_CACHE_SEC:
            return {
                "enabled": True,
                "emptied": False,
                "idle_for_sec": round(idle_for, 2),
                "threshold_sec": IDLE_EMPTY_CACHE_SEC,
            }

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                "TotalSeg idle empty_cache (idle_for=%.1fs, threshold=%.1fs)",
                idle_for,
                IDLE_EMPTY_CACHE_SEC,
            )
            return {
                "enabled": True,
                "emptied": True,
                "idle_for_sec": round(idle_for, 2),
                "threshold_sec": IDLE_EMPTY_CACHE_SEC,
            }
    except Exception as e:
        logger.warning("TotalSeg idle empty_cache failed: %s", e)
    return {"enabled": True, "emptied": False, "error": True}


def ensure_totalseg_idle_reaper_started() -> None:
    global _reaper_started
    if _reaper_started or IDLE_EMPTY_CACHE_SEC <= 0:
        return
    with _lock:
        if _reaper_started:
            return

        def _loop() -> None:
            while True:
                try:
                    time.sleep(max(1.0, IDLE_CHECK_SEC))
                    run_totalseg_idle_empty_cache_check()
                except Exception as e:
                    logger.warning("TotalSeg idle reaper loop error: %s", e)

        t = threading.Thread(target=_loop, name="totalseg-idle-reaper", daemon=True)
        t.start()
        _reaper_started = True
        logger.info(
            "TotalSeg idle empty_cache enabled (sec=%s, check=%ss)",
            IDLE_EMPTY_CACHE_SEC,
            IDLE_CHECK_SEC,
        )


def idle_policy_snapshot() -> dict[str, Any]:
    return {
        "totalseg_idle_empty_cache_sec": IDLE_EMPTY_CACHE_SEC,
        "totalseg_idle_check_sec": IDLE_CHECK_SEC,
        "reaper_enabled": IDLE_EMPTY_CACHE_SEC > 0,
    }
