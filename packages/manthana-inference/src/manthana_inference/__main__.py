"""Parse cloud_inference.yaml (no network). Run: python -m manthana_inference"""

from __future__ import annotations

import sys

from manthana_inference.loader import load_cloud_inference_config


def main() -> int:
    try:
        cfg = load_cloud_inference_config()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: schema_version={cfg.schema_version} roles={len(cfg.roles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
