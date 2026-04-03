#!/usr/bin/env python3
"""
One-time / pre-deploy migration for on-disk model registry.json.

Resolves REGISTRY_PATH the same way as shared/model_registry.py:
  MODEL_REGISTRY_PATH, or $MODEL_DIR/registry.json, default /models/registry.json.

Legacy keys (pre–CXR pipeline lock):
  medrax     → txrv_primary
  evax       → txrv_secondary
  chexagent  → removed (no replacement)

Safe to run repeatedly (idempotent). Safe if the file is missing (no-op).

CI / deploy: run before starting services that read the registry, e.g.
  cd manthana-backend && python3 scripts/migrate_model_registry.py
  MODEL_REGISTRY_PATH=/data/registry.json python3 scripts/migrate_model_registry.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "shared"))

from model_registry import REGISTRY_PATH  # noqa: E402

MIGRATIONS: dict[str, str | None] = {
    "medrax": "txrv_primary",
    "evax": "txrv_secondary",
    "chexagent": None,
}


def main() -> int:
    path = REGISTRY_PATH
    print(f"Registry path: {path}")

    if not os.path.isfile(path):
        print("No registry file on disk — nothing to migrate.")
        return 0

    with open(path, encoding="utf-8") as f:
        reg = json.load(f)

    if not isinstance(reg, dict):
        print("registry.json is not a JSON object — aborting.", file=sys.stderr)
        return 1

    changed = False

    for old_key, new_key in MIGRATIONS.items():
        if old_key not in reg:
            continue
        if new_key is None:
            del reg[old_key]
            print(f"Deleted:  {old_key}")
            changed = True
            continue
        if new_key in reg:
            del reg[old_key]
            print(
                f"Removed legacy key {old_key!r} "
                f"(target {new_key!r} already present)"
            )
        else:
            reg[new_key] = reg.pop(old_key)
            print(f"Migrated: {old_key} → {new_key}")
        changed = True

    if changed:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
        print("registry.json updated.")
    else:
        print("Nothing to migrate — registry already clean.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
