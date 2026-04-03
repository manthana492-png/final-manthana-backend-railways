from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "gateway"))
sys.path.insert(0, str(BACKEND_ROOT / "shared"))
sys.path.insert(0, str(BACKEND_ROOT / "services" / "01_body_xray"))
sys.path.insert(0, str(BACKEND_ROOT / "services" / "report_assembly"))
sys.path.insert(0, str(BACKEND_ROOT / "services" / "15_lab_report"))


@pytest.fixture
def sample_xray_path(tmp_path: Path) -> str:
    """Single-channel-like random PNG (saved as grayscale)."""
    path = tmp_path / "sample_gray.png"
    arr = (np.random.rand(256, 256) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture
def sample_rgb_xray_path(tmp_path: Path) -> str:
    """RGB random PNG (tests channel mean → gray)."""
    path = tmp_path / "sample_rgb.png"
    arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)
