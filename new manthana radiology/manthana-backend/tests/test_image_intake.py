"""Tests for shared image_intake (format detection, merge helpers)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from image_intake import (
    _detect_format,
    intake_pil_to_temp_path,
    merge_image_quality_into_result,
    normalize_for_model,
    raw_to_pil,
)


def test_detect_format_png_jpeg():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    assert _detect_format(png) == "png"
    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 20
    assert _detect_format(jpeg) == "jpeg"


def test_raw_to_pil_roundtrip_rgb():
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()
    pil = raw_to_pil(raw)
    assert pil.size == (32, 32)


def test_normalize_for_model_xray_returns_quality():
    img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = normalize_for_model(buf.getvalue(), modality="xray")
    assert out["array"] is None
    assert out["pil"] is not None
    assert "resolution" in out["quality"]
    assert "warnings" in out["quality"]
    assert "quality_ok" in out["quality"]


def test_merge_image_quality_into_result():
    r = {
        "structures": {"narrative_report": "x", "pathology_fractions": {}},
    }
    q = {"resolution": "100×100", "warnings": ["low"], "quality_ok": False}
    merge_image_quality_into_result(r, q)
    assert r["structures"]["image_quality"] == q


def test_intake_pil_to_temp_path(tmp_path):
    img = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
    p = intake_pil_to_temp_path(img, str(tmp_path), "job1", prefer_grayscale=True)
    assert Path(p).is_file()
    assert p.endswith("job1.png")
