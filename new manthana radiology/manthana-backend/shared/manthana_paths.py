"""Resolve ``manthana-backend`` root for services in repo layout vs Modal flat ``/app`` layout.

Services are copied to ``/app`` in Modal images (``main.py`` at ``/app/main.py``). Repo layout is
``manthana-backend/services/<id>/main.py``. Path parents differ; use this helper instead of
``Path(__file__).parents[N]``.
"""

from __future__ import annotations

import os
from pathlib import Path


def backend_root_from_service_file(service_file: Path | str) -> Path:
    """Directory that contains ``shared/`` and ``services/`` (Modal: ``/app``)."""
    f = Path(service_file).resolve()
    env = os.environ.get("MANTHANA_BACKEND_ROOT")
    if env:
        return Path(env)
    if Path("/app/shared").is_dir():
        try:
            f.relative_to(Path("/app"))
            return Path("/app")
        except ValueError:
            pass
    return f.parents[2]
