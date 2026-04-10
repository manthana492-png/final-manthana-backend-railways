"""
Manthana — JWT Authentication
Verify bearer tokens at the gateway level.

Supports:
- Supabase access tokens (HS256, aud=authenticated) when SUPABASE_JWT_SECRET is set
- Legacy gateway-minted tokens via JWT_SECRET (pilot / auth_routes)
"""

import os
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-to-a-random-secret-minimum-32-chars")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
SUPABASE_JWT_ISS = os.getenv("SUPABASE_JWT_ISS", "").strip()

security = HTTPBearer(auto_error=False)


def create_token(user_id: str, role: str = "doctor",
                 extra: dict = None) -> str:
    """Create a JWT token for a user (legacy gateway minting)."""
    payload: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    if extra:
        payload.update(extra)

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_supabase(token: str) -> dict:
    options: dict[str, Any] = {
        "algorithms": ["HS256"],
        "audience": "authenticated",
    }
    if SUPABASE_JWT_ISS:
        options["issuer"] = SUPABASE_JWT_ISS
    return jwt.decode(token, SUPABASE_JWT_SECRET, **options)


def _decode_legacy(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Verify JWT from Authorization header.

    If SUPABASE_JWT_SECRET is set, tries Supabase access token verification first
    (HS256, audience ``authenticated``), then falls back to JWT_SECRET.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials

    if SUPABASE_JWT_SECRET:
        try:
            return _decode_supabase(token)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired") from None
        except jwt.InvalidAudienceError:
            pass
        except jwt.InvalidTokenError:
            pass

    try:
        return _decode_legacy(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired") from None
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e
