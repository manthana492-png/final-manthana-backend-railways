"""Optional GPU VRAM hygiene for CT Brain TorchScript inference (empty_cache after idle)."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("manthana.ct_brain_gpu_idle")

IDLE_EMPTY_CACHE_SEC = float(os.getenv("CT_BRAIN_GPU_IDLE_EMPTY_CACHE_SEC", "0") or "0")
IDLE_CHECK_SEC = float(os.getenv("CT_BRAIN_GPU_IDLE_CHECK_SEC", "30") or "30")

_last_activity_monotonic: float = 0.0
_lock = threading.Lock()
_reaper_started = False


def touch_ct_brain_gpu_activity() -> None:
    global _last_activity_monotonic
    with _lock:
        _last_activity_monotonic = time.monotonic()


def run_ct_brain_idle_empty_cache_check() -> dict[str, Any]:
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
                "CT Brain idle empty_cache (idle_for=%.1fs, threshold=%.1fs)",
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
        logger.warning("CT Brain idle empty_cache failed: %s", e)
    return {"enabled": True, "emptied": False, "error": True}


def ensure_ct_brain_idle_reaper_started() -> None:
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
                    run_ct_brain_idle_empty_cache_check()
                except Exception as e:
                    logger.warning("CT Brain idle reaper loop error: %s", e)

        t = threading.Thread(target=_loop, name="ct-brain-idle-reaper", daemon=True)
        t.start()
        _reaper_started = True
        logger.info(
            "CT Brain idle empty_cache enabled (sec=%s, check=%ss)",
            IDLE_EMPTY_CACHE_SEC,
            IDLE_CHECK_SEC,
        )


def idle_policy_snapshot() -> dict[str, Any]:
    return {
        "ct_brain_idle_empty_cache_sec": IDLE_EMPTY_CACHE_SEC,
        "ct_brain_idle_check_sec": IDLE_CHECK_SEC,
        "reaper_enabled": IDLE_EMPTY_CACHE_SEC > 0,
    }
