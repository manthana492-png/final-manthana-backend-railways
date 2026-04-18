"""DPDP-oriented audit log: JSON lines, no PHI or image content."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

_audit_logger = logging.getLogger("manthana.audit")


def _setup_audit_logging() -> None:
    if _audit_logger.handlers:
        return
    path = (os.environ.get("AUDIT_LOG_PATH") or "audit.log").strip() or "audit.log"
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.addHandler(handler)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False


_setup_audit_logging()


def log_analysis_event(
    user_id: str,
    event_type: str,
    modality_key: Optional[str] = None,
    group: Optional[str] = None,
    subscription_tier: str = "unknown",
    model_used: Optional[str] = None,
    success: bool = True,
    error_code: Optional[int] = None,
    image_mime: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Log who did what (metadata only). Never pass image_b64 or patient text."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uid": user_id,
        "event": event_type,
        "modality": modality_key,
        "group": group,
        "tier": subscription_tier,
        "model": model_used,
        "ok": success,
        "mime_type": image_mime,
        "session": session_id,
        "err": error_code,
    }
    _audit_logger.info(json.dumps(record, separators=(",", ":")))
