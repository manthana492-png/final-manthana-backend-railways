"""Load key=value pairs from api-keys.env into os.environ (setdefault)."""

from __future__ import annotations

import os
from pathlib import Path


def load_api_keys_env() -> None:
    candidates: list[Path] = [
        Path("/teamspace/studios/this_studio/api-keys.env"),
    ]
    here = Path(__file__).resolve()
    p = here.parent
    for _ in range(16):
        candidates.append(p / "api-keys.env")
        if p.parent == p:
            break
        p = p.parent
    candidates.append(Path.cwd() / "api-keys.env")
    candidates.append(Path.cwd().parent / "api-keys.env")

    for p in candidates:
        if not p.is_file():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        except OSError:
            continue
        break
