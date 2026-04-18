"""Per-user rate limit for /ai/interrogate (single-replica; Redis TODO for scale-out)."""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException

_FREE_LIMIT = int(os.environ.get("AI_INTERROGATE_RATE_LIMIT_FREE", "10"))
_PAID_LIMIT = int(os.environ.get("AI_INTERROGATE_RATE_LIMIT_PAID", "100"))
_WINDOW = int(os.environ.get("AI_RATE_WINDOW_SECONDS", "3600"))

FREE_TIERS = frozenset({"free", "trial", ""})


class InMemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._store: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check_and_record(self, user_id: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            timestamps = self._store[user_id]
            self._store[user_id] = [t for t in timestamps if now - t < self.window]
            if len(self._store[user_id]) >= self.max_requests:
                oldest = self._store[user_id][0]
                retry_after = int(self.window - (now - oldest)) + 1
                return False, retry_after
            self._store[user_id].append(now)
            return True, 0


_free_limiter = InMemoryRateLimiter(_FREE_LIMIT, _WINDOW)
_paid_limiter = InMemoryRateLimiter(_PAID_LIMIT, _WINDOW)


def enforce_rate_limit(user_id: str, subscription_tier: str) -> None:
    tier = (subscription_tier or "free").strip().lower()
    limiter = _free_limiter if tier in FREE_TIERS else _paid_limiter
    allowed, retry_after = limiter.check_and_record(user_id)
    if not allowed:
        lim = _FREE_LIMIT if tier in FREE_TIERS else _PAID_LIMIT
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded. Please wait before submitting another analysis.",
                "retry_after_seconds": retry_after,
                "limit_per_hour": lim,
            },
            headers={"Retry-After": str(retry_after)},
        )
