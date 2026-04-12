"""
Manthana — JWT Authentication
Verify bearer tokens at the gateway level.

Supports:
- Supabase access tokens:
  - Asymmetric (RS256, ES256, …) via JWKS — matches current Supabase guidance when
    JWT Signing Keys are enabled (see Supabase docs: auth/v1/.well-known/jwks.json).
  - Symmetric HS256 with SUPABASE_JWT_SECRET (legacy / shared-secret signing).
- Legacy gateway-minted tokens via JWT_SECRET (pilot / auth_routes).
"""

import os
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt.exceptions import PyJWKClientError

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-to-a-random-secret-minimum-32-chars")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
SUPABASE_JWT_ISS = os.getenv("SUPABASE_JWT_ISS", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL", "").strip()

_ASYM_ALGS = frozenset(
    {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "EdDSA", "Ed25519"}
)

security = HTTPBearer(auto_error=False)


def _supabase_issuer() -> Optional[str]:
    if SUPABASE_JWT_ISS:
        return SUPABASE_JWT_ISS.rstrip("/")
    if SUPABASE_URL:
        return f"{SUPABASE_URL.rstrip('/')}/auth/v1"
    return None


def _jwks_url() -> Optional[str]:
    if SUPABASE_JWKS_URL:
        return SUPABASE_JWKS_URL.rstrip("/")
    iss = _supabase_issuer()
    if iss:
        return f"{iss}/.well-known/jwks.json"
    if SUPABASE_URL:
        return f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    return None


def _supabase_configured() -> bool:
    return bool(SUPABASE_JWT_SECRET or _jwks_url())


def _token_issuer(token: str) -> Optional[str]:
    try:
        iss = _unverified_claims(token).get("iss")
    except jwt.DecodeError:
        return None
    if not isinstance(iss, str) or not iss:
        return None
    return iss.rstrip("/")


def _jwks_url_from_token_issuer(token: str) -> Optional[str]:
    iss = _token_issuer(token)
    if not iss:
        return None
    if "supabase.co" not in iss:
        return None
    if iss.endswith("/auth/v1"):
        return f"{iss}/.well-known/jwks.json"
    return None


def _unverified_claims(token: str) -> dict[str, Any]:
    return jwt.decode(
        token,
        options={
            "verify_signature": False,
            "verify_aud": False,
            "verify_exp": False,
        },
        algorithms=["HS256", "RS256", "ES256", "ES384", "RS512", "EdDSA"],
    )


def _looks_like_supabase_access_token(token: str) -> bool:
    """Heuristic so failed Supabase verification returns specific errors, not legacy 'Invalid token'.

    Custom Auth domains often omit 'supabase.co' in ``iss``; match configured ``SUPABASE_JWT_ISS``
    / derived issuer from ``SUPABASE_URL`` so JWKS/issuer checks still map to Supabase errors.
    """
    try:
        claims = _unverified_claims(token)
        iss = claims.get("iss")
        if isinstance(iss, str) and iss:
            if "supabase.co" in iss:
                return True
            cfg_iss = _supabase_issuer()
            if cfg_iss and iss.rstrip("/") == cfg_iss.rstrip("/"):
                return True
        aud = claims.get("aud")
        if aud == "authenticated":
            return True
        if isinstance(aud, list) and "authenticated" in aud:
            return True
    except jwt.DecodeError:
        pass
    return False


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


def _decode_supabase_symmetric(token: str) -> dict:
    options: dict[str, Any] = {
        "algorithms": ["HS256"],
        "audience": "authenticated",
    }
    iss = _supabase_issuer()
    if iss:
        options["issuer"] = iss
    return jwt.decode(token, SUPABASE_JWT_SECRET, **options)


def _decode_supabase_asymmetric(token: str, alg: str) -> dict:
    jwks = _jwks_url() or _jwks_url_from_token_issuer(token)
    if not jwks:
        raise jwt.InvalidTokenError(
            "Asymmetric Supabase JWT (alg=%s) requires SUPABASE_JWKS_URL, "
            "SUPABASE_JWT_ISS, SUPABASE_URL, or a Supabase issuer in token 'iss' "
            "for JWKS lookup" % alg
        )
    jwk_client = jwt.PyJWKClient(jwks, cache_keys=True)
    signing_key = jwk_client.get_signing_key_from_jwt(token)
    options: dict[str, Any] = {
        "algorithms": [alg],
        "audience": "authenticated",
    }
    iss = _supabase_issuer()
    if iss:
        options["issuer"] = iss
    return jwt.decode(token, signing_key.key, **options)


def _decode_supabase(token: str) -> dict:
    """Verify Supabase-issued access token (asymmetric JWKS or HS256 secret)."""
    header = jwt.get_unverified_header(token)
    alg = (header.get("alg") or "").upper()

    if alg in _ASYM_ALGS:
        return _decode_supabase_asymmetric(token, alg)

    if SUPABASE_JWT_SECRET:
        return _decode_supabase_symmetric(token)

    raise jwt.InvalidTokenError(
        "Unrecognized JWT algorithm for Supabase verification: %s "
        "(set SUPABASE_JWT_SECRET for HS256 or SUPABASE_URL / SUPABASE_JWT_ISS for JWKS)"
        % (alg or "(missing)")
    )


def _decode_legacy(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def _supabase_failure_detail(exc: Exception) -> str:
    if isinstance(exc, jwt.ExpiredSignatureError):
        return "Token expired"
    if isinstance(exc, jwt.InvalidAudienceError):
        return "Invalid Supabase token (audience must include 'authenticated')"
    if isinstance(exc, jwt.InvalidIssuerError):
        return "Invalid Supabase token (issuer mismatch — check SUPABASE_JWT_ISS or SUPABASE_URL)"
    if isinstance(exc, jwt.InvalidSignatureError):
        return "Invalid Supabase token (signature — HS256 secret wrong or JWKS out of sync)"
    msg = getattr(exc, "args", [None])[0]
    if isinstance(msg, str) and msg:
        return f"Invalid Supabase token ({msg})"
    return "Invalid Supabase token"


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Verify JWT from Authorization header.

    Tries Supabase verification when ``SUPABASE_JWT_SECRET`` and/or JWKS URL is
    available (JWKS derived from ``SUPABASE_JWT_ISS`` or ``SUPABASE_URL`` if unset),
    then legacy ``JWT_SECRET`` tokens.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials

    should_try_supabase = _supabase_configured() or _looks_like_supabase_access_token(token)

    if should_try_supabase:
        try:
            return _decode_supabase(token)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired") from None
        except PyJWKClientError as e:
            if _looks_like_supabase_access_token(token):
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid Supabase token (JWKS: {e!s})",
                ) from None
        except jwt.InvalidTokenError as e:
            if _looks_like_supabase_access_token(token):
                raise HTTPException(
                    status_code=401,
                    detail=_supabase_failure_detail(e),
                ) from None
            # Likely a legacy gateway token — try below

    try:
        return _decode_legacy(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired") from None
    except jwt.InvalidTokenError as e:
        if should_try_supabase and _looks_like_supabase_access_token(token):
            raise HTTPException(
                status_code=401,
                detail=_supabase_failure_detail(e),
            ) from None
        if should_try_supabase:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid token: not verified as Supabase (issuer/audience mismatch or wrong "
                    "algorithm path) and not as legacy gateway JWT. Align SUPABASE_JWT_ISS with "
                    "the token iss claim, set SUPABASE_JWT_SECRET or JWKS per your project, and "
                    "ensure the browser sends the session access_token."
                ),
            ) from e
        raise HTTPException(status_code=401, detail="Invalid token") from e
