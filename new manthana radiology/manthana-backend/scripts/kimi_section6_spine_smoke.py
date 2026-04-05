#!/usr/bin/env python3
"""Section 6 smoke test: spine / Pott's narrative via OpenRouter (role spine, SSOT config/cloud_inference.yaml).

Requires OPENROUTER_API_KEY. Optional: CLOUD_INFERENCE_CONFIG_PATH to repo root YAML.
"""
from __future__ import annotations

import os
import sys

_BACKEND = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.join(_BACKEND, "shared")
_REPO_ROOT = os.path.normpath(os.path.join(_BACKEND, "..", ".."))


def _load_env_file() -> None:
    for path in (
        os.path.join(_REPO_ROOT, "api-keys.env"),
        os.path.join(_BACKEND, ".env"),
    ):
        path = os.path.normpath(path)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    _load_env_file()
    if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        print("Set OPENROUTER_API_KEY", file=sys.stderr)
        return 1
    if not os.environ.get("CLOUD_INFERENCE_CONFIG_PATH"):
        yaml_path = os.path.join(_REPO_ROOT, "config", "cloud_inference.yaml")
        if os.path.isfile(yaml_path):
            os.environ["CLOUD_INFERENCE_CONFIG_PATH"] = yaml_path

    if _SHARED not in sys.path:
        sys.path.insert(0, _SHARED)

    try:
        from llm_router import llm_router
    except Exception as e:
        print(f"llm_router import failed: {e}", file=sys.stderr)
        return 1

    out = llm_router.complete_for_role(
        "spine",
        "You are a senior neuroradiologist.",
        (
            "Pott's disease L2-L3, paravertebral abscess, 35M Bihar. "
            "200 word structured report with RNTCP protocol. "
            "Explicitly recommend CT-guided biopsy for AFB."
        ),
        max_tokens=4096,
    )
    text = (out.get("content") or "").strip()
    print(text)
    required = ["Pott", "L2", "L3", "abscess", "biopsy", "TB", "RNTCP"]
    missing = [t for t in required if t.lower() not in text.lower()]
    if missing:
        print(f"WARN missing: {missing}", file=sys.stderr)
        return 2
    print("PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
