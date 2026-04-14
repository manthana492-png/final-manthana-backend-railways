"""TTL session store for /ai/interrogate -> /ai/interpret."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional

_DEFAULT_TTL = int(__import__("os").getenv("AI_SESSION_TTL_SECONDS", "1800"))


class SessionStore:
    def __init__(self, ttl_seconds: int = _DEFAULT_TTL) -> None:
        self._ttl = ttl_seconds
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        image_b64: Optional[str],
        image_mime: str,
        modality_key: str,
        display_name: str,
        group: str,
        questions: List[Dict[str, Any]],
        interrogator_model: str,
        interpreter_role: str,
        patient_context_json: Optional[str] = None,
    ) -> str:
        sid = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._cleanup_unlocked(now)
            self._data[sid] = {
                "created_at": now,
                "image_b64": image_b64,
                "image_mime": image_mime or "image/jpeg",
                "modality_key": modality_key,
                "display_name": display_name,
                "group": group,
                "questions": questions,
                "interrogator_model": interrogator_model,
                "interpreter_role": interpreter_role,
                "patient_context_json": patient_context_json,
            }
        return sid

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            self._cleanup_unlocked(now)
            row = self._data.get(session_id)
            if not row:
                return None
            if now - float(row["created_at"]) > self._ttl:
                self._data.pop(session_id, None)
                return None
            return dict(row)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def _cleanup_unlocked(self, now: float) -> None:
        dead = [k for k, v in self._data.items() if now - float(v["created_at"]) > self._ttl]
        for k in dead:
            self._data.pop(k, None)


session_store = SessionStore()
