"""DPDP-style consent logging with JSONL persistence."""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import verify_token

router = APIRouter()

CONSENT_LOG_PATH = pathlib.Path(
    os.getenv("CONSENT_LOG_PATH", "/app/data/consent_log.jsonl")
)


class ConsentRequest(BaseModel):
    patient_id: str
    consent_version: str = "v1.0"
    purpose: str
    informed_by: str


@router.post("/consent")
def record_consent(
    req: ConsentRequest,
    token_data: dict = Depends(verify_token),
) -> dict:
    record = {
        "patient_id": req.patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "consent_version": req.consent_version,
        "purpose": req.purpose,
        "informed_by": req.informed_by,
    }
    try:
        CONSENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONSENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist consent record: {e}",
        )
    return {"status": "recorded", "record": record}


@router.get("/consent/log")
def get_consent_log(token_data: dict = Depends(verify_token)) -> dict:
    """
    Admin-only view of consent records.
    Reads JSONL file and returns all records; empty if file is missing.
    """
    if token_data.get("role") not in {"admin", "superadmin"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if not CONSENT_LOG_PATH.is_file():
        return {"records": [], "count": 0}

    records: List[dict] = []
    try:
        with CONSENT_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read consent log: {e}",
        )

    return {"records": records, "count": len(records)}
