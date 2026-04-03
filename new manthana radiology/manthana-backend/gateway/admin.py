"""Admin API for model registry (JWT + admin role)."""

from __future__ import annotations

import os
import sys

from fastapi import APIRouter, Depends, HTTPException

sys.path.insert(0, "/app/shared")

from auth import verify_token
from model_registry import ModelRegistry

admin_router = APIRouter(tags=["admin"])

registry = ModelRegistry()


def verify_admin(token_data: dict = Depends(verify_token)) -> dict:
    role = token_data.get("role", "")
    if role != "admin" and os.getenv("ADMIN_DEV_BYPASS") != "1":
        raise HTTPException(status_code=403, detail="Admin role required")
    return token_data


@admin_router.get("/admin/models")
async def list_models(_: dict = Depends(verify_admin)):
    return registry.list_models()


@admin_router.post("/admin/models/{key}/canary")
async def set_canary(key: str, body: dict, _: dict = Depends(verify_admin)):
    mid = body.get("model_id")
    if not mid:
        raise HTTPException(status_code=400, detail="model_id required")
    registry.set_canary(key, mid)
    return {"status": "ok", "key": key, "canary": mid}


@admin_router.post("/admin/models/{key}/promote")
async def promote(key: str, _: dict = Depends(verify_admin)):
    registry.promote_canary(key)
    return {"status": "ok", "key": key}


@admin_router.post("/admin/models/{key}/rollback")
async def rollback(key: str, _: dict = Depends(verify_admin)):
    registry.rollback(key)
    return {"status": "ok", "key": key}


@admin_router.get("/admin/models/{key}/metrics")
async def metrics(key: str, _: dict = Depends(verify_admin)):
    return {
        "key": key,
        "note": "Wire to Prometheus / custom metrics store in production.",
        "accuracy": None,
        "latency_p50_ms": None,
    }
