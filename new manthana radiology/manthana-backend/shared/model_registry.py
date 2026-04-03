"""Manthana — Model Registry (versions, canary, rollback).

Deploys with an existing on-disk registry.json should run
``scripts/migrate_model_registry.py`` once (or in CI pre-deploy) to rename
legacy keys (medrax/evax) to txrv_primary/txrv_secondary and drop chexagent.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from typing import Any

logger = logging.getLogger("manthana.model_registry")

_MODEL_DIR = os.getenv("MODEL_DIR") or os.getenv("MANTHANA_MODEL_CACHE", "/tmp/manthana_models")
REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", os.path.join(_MODEL_DIR, "registry.json"))


def _default_entries() -> dict[str, Any]:
    return {
        "txrv_primary": {
            "current": {"model_id": "densenet121-res224-all"},
            "previous": {"model_id": "densenet121-res224-all"},
        },
        "txrv_secondary": {
            "current": {"model_id": "densenet121-res224-chex"},
            "previous": {"model_id": "densenet121-res224-chex"},
        },
        "parrotlet-v-lite-4b": {
            "current": {"model_id": "ekacare/parrotlet-v-lite-4b"},
            "previous": {"model_id": "ekacare/parrotlet-v-lite-4b"},
        },
        "parrotlet-e": {
            "current": {"model_id": "ekacare/parrotlet-e"},
            "previous": {"model_id": "ekacare/parrotlet-e"},
        },
    }


class ModelRegistry:
    """Tracks HuggingFace model IDs per logical key."""

    def __init__(self):
        self._lock = threading.Lock()
        self.entries = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        if os.path.isfile(REGISTRY_PATH):
            try:
                with open(REGISTRY_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to read registry: %s", e)
        data = _default_entries()
        parent = os.path.dirname(REGISTRY_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    def _save(self, data: dict[str, Any] | None = None) -> None:
        data = data or self.entries
        parent = os.path.dirname(REGISTRY_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_active_model_id(self, model_key: str) -> str:
        with self._lock:
            entry = self.entries.get(model_key)
            if not entry:
                raise KeyError(model_key)
            if entry.get("canary") and random.random() < 0.05:
                return entry["canary"]["model_id"]
            return entry["current"]["model_id"]

    def rollback(self, model_key: str) -> None:
        with self._lock:
            entry = self.entries[model_key]
            cur = entry.get("current")
            prev = entry.get("previous")
            if cur and prev:
                entry["current"], entry["previous"] = prev, cur
            self._save()

    def promote_canary(self, model_key: str) -> None:
        with self._lock:
            entry = self.entries.get(model_key)
            if not entry or "canary" not in entry:
                return
            entry["previous"] = entry["current"]
            entry["current"] = entry.pop("canary")
            self._save()

    def set_canary(self, model_key: str, model_id: str) -> None:
        with self._lock:
            if model_key not in self.entries:
                self.entries[model_key] = {
                    "current": {"model_id": model_id},
                    "previous": {"model_id": model_id},
                }
            else:
                self.entries[model_key]["canary"] = {"model_id": model_id}
            self._save()

    def list_models(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.entries))
