#!/usr/bin/env python3
"""Section 6 smoke test: Kimi spine / Pott's narrative (run when API is not rate-limited).

Uses KIMI_BASE_URL from api-keys.env (typically https://api.moonshot.ai/v1).
If you use https://api.moonshot.cn/v1, ensure your key is issued for that region.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_ENV = "/teamspace/studios/this_studio/api-keys.env"


def _load_env() -> None:
    path = _ENV if os.path.isfile(_ENV) else os.path.join(_ROOT, "..", "..", "api-keys.env")
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def main() -> int:
    _load_env()
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package required", file=sys.stderr)
        return 1

    base = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    kimi = OpenAI(api_key=os.environ["KIMI_API_KEY"], base_url=base)
    model = os.environ.get("KIMI_SPINE_MODEL", os.environ.get("KIMI_MODEL", "moonshot-v1-8k"))
    r = kimi.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Senior neuroradiologist."},
            {
                "role": "user",
                "content": (
                    "Pott's disease L2-L3, paravertebral abscess, 35M Bihar. "
                    "200 word structured report with RNTCP protocol. "
                    "Explicitly recommend CT-guided biopsy for AFB."
                ),
            },
        ],
        max_tokens=4096,
    )
    text = (r.choices[0].message.content or "").strip()
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
