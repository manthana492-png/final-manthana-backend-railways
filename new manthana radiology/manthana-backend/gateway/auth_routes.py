from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi import Depends
from pydantic import BaseModel
import jwt

from auth import JWT_SECRET, JWT_ALGORITHM, verify_token


router = APIRouter()


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _load_pilot_users() -> List[Dict[str, Any]]:
    raw = os.getenv(
        "PILOT_USERS_JSON",
        '[{"username":"radiologist1","password":"changeme","role":"radiologist"}]',
    )
    try:
        users = json.loads(raw)
        if raw.strip().startswith('[{"username":"radiologist1"'):
            # Default config still in use
            import logging

            logging.getLogger("manthana.gateway").warning(
                "PILOT_USERS_JSON is using the default demo credentials. "
                "Update this env var before production use."
            )
        if isinstance(users, list):
            return [u for u in users if isinstance(u, dict)]
    except json.JSONDecodeError:
        pass
    return []


PILOT_USERS = _load_pilot_users()


@router.post("/auth/token", response_model=TokenResponse)
async def issue_token(req: TokenRequest) -> TokenResponse:
    """
    Minimal JWT issuance endpoint for pilot use.
    Validates against a static user store from PILOT_USERS_JSON.
    """
    user = next((u for u in PILOT_USERS if u.get("username") == req.username), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    stored_pw = str(user.get("password", ""))
    if not secrets.compare_digest(stored_pw, req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    role = str(user.get("role", "radiologist"))
    hours = int(os.getenv("JWT_EXPIRY_HOURS", "12"))
    now = datetime.utcnow()
    exp = now + timedelta(hours=hours)

    payload: Dict[str, Any] = {
        "sub": req.username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return TokenResponse(
        access_token=token,
        expires_in=int((exp - now).total_seconds()),
    )


@router.get("/auth/me")
async def auth_me(token_data: dict = Depends(verify_token)) -> dict:
    """
    Return the decoded JWT payload for the current user.
    Useful for frontend to confirm login state and role.
    """
    return token_data

