"""In-process session store with TTL (Modal single-container; use Redis later if scaled)."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CxrMedgemmaSession:
    session_id: str
    created_at: float
    image_path: str
    pathology_scores: Dict[str, Any]
    patient_context: Dict[str, Any]
    medgemma_stage1: Dict[str, Any] = field(default_factory=dict)
    answers: Dict[str, str] = field(default_factory=dict)


class SessionStore:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: Dict[str, CxrMedgemmaSession] = {}

    def _purge_locked(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items() if now - s.created_at > self._ttl]
        for sid in dead:
            self._sessions.pop(sid, None)

    def create(
        self,
        image_path: str,
        pathology_scores: Dict[str, Any],
        patient_context: Dict[str, Any],
        medgemma_stage1: Dict[str, Any],
    ) -> CxrMedgemmaSession:
        sid = secrets.token_urlsafe(24)
        sess = CxrMedgemmaSession(
            session_id=sid,
            created_at=time.time(),
            image_path=image_path,
            pathology_scores=pathology_scores,
            patient_context=patient_context,
            medgemma_stage1=medgemma_stage1,
        )
        with self._lock:
            self._purge_locked()
            self._sessions[sid] = sess
        return sess

    def get(self, session_id: str) -> Optional[CxrMedgemmaSession]:
        with self._lock:
            self._purge_locked()
            s = self._sessions.get(session_id)
            if s is None:
                return None
            if time.time() - s.created_at > self._ttl:
                self._sessions.pop(session_id, None)
                return None
            return s

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
