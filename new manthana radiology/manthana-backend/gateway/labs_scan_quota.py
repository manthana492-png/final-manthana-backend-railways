"""Manthana Labs: lifetime completed-report cap for free-tier users (orchestration /interpret)."""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict

from fastapi import HTTPException

logger = logging.getLogger("manthana.labs_scan_quota")

_FREE_TIERS = frozenset({"free", "trial", ""})

_limit = max(0, int((os.environ.get("AI_LABS_FREE_LIFETIME_SCANS") or "3").strip() or "3"))
_persist_path = (os.environ.get("AI_LABS_LIFETIME_QUOTA_PATH") or "").strip()

_lock = threading.Lock()
_counts: Dict[str, int] = {}


def _tier_is_free(tier: str) -> bool:
    return (tier or "free").strip().lower() in _FREE_TIERS


def _load() -> None:
    global _counts
    if not _persist_path:
        return
    try:
        with open(_persist_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            out: Dict[str, int] = {}
            for k, v in raw.items():
                try:
                    out[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
            _counts = out
    except FileNotFoundError:
        _counts = {}
    except Exception as e:
        logger.warning("labs lifetime quota load failed: %s", e)


def _save() -> None:
    if not _persist_path:
        return
    try:
        with open(_persist_path, "w", encoding="utf-8") as f:
            json.dump(_counts, f, indent=0)
    except Exception as e:
        logger.warning("labs lifetime quota save failed: %s", e)


_load()


def lifetime_interpret_count(user_id: str) -> int:
    with _lock:
        return int(_counts.get(user_id, 0))


def record_interpret_success(user_id: str, tier: str) -> None:
    """Count one completed Labs report for free-tier users only."""
    if _limit <= 0 or not _tier_is_free(tier):
        return
    with _lock:
        _counts[user_id] = int(_counts.get(user_id, 0)) + 1
        _save()


def assert_labs_lifetime_quota(user_id: str, tier: str) -> None:
    """Block new interrogate/interpret when free tier has used all lifetime scans."""
    if _limit <= 0 or not _tier_is_free(tier):
        return
    with _lock:
        used = int(_counts.get(user_id, 0))
    if used >= _limit:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "labs_lifetime_quota_exceeded",
                "message": (
                    f"Manthana Labs includes {_limit} full analyses per account on the free tier. "
                    "Upgrade for unlimited scans with the same clinical-grade models."
                ),
                "lifetime_scans_used": used,
                "lifetime_scans_limit": _limit,
            },
        )
