"""ECG pipeline: schema, rhythm exclusion, PDF rejection, correlation keys."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "shared"))
sys.path.insert(0, str(BACKEND / "services" / "13_ecg"))
sys.path.insert(0, str(BACKEND / "services" / "report_assembly"))


def _load_shared_schemas():
    """Avoid collision with gateway/schemas.py when tests import gateway first."""
    path = BACKEND / "shared" / "schemas.py"
    spec = importlib.util.spec_from_file_location("manthana_shared_schemas", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_rhythm_keys_for_correlation():
    from ecg_rhythm import RHYTHM_KEYS

    assert "sinus_rhythm" in RHYTHM_KEYS
    for key in ("atrial_fibrillation", "lvh", "lbbb"):
        assert key in RHYTHM_KEYS


def test_empty_rhythm_scores_are_zero():
    from ecg_rhythm import rhythm_scores_from_signal, RHYTHM_KEYS

    out = rhythm_scores_from_signal(np.array([]).reshape(0, 0), 500.0)
    assert all(out[k] == 0.0 for k in RHYTHM_KEYS)


def test_pdf_rejected_not_image():
    from preprocessing.ecg_utils import detect_input_type
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
        f.write(b"%PDF-1.4")
    try:
        assert detect_input_type(path) == "pdf_rejected"
    finally:
        os.unlink(path)


def test_significant_excludes_sinus_rhythm():
    from inference import _significant_abnormalities

    scores = {"sinus_rhythm": 0.95, "atrial_fibrillation": 0.1}
    sig = _significant_abnormalities(scores)
    assert "sinus_rhythm" not in sig


def test_analysis_response_structures_dict():
    mod = _load_shared_schemas()
    AnalysisResponse = mod.AnalysisResponse
    Finding = mod.Finding

    f = Finding(
        label="AF",
        description="test",
        severity="warning",
        confidence=60.0,
        region="cardiac",
    )
    r = AnalysisResponse(
        job_id="x",
        modality="ecg",
        findings=[f],
        impression="imp",
        pathology_scores={"atrial_fibrillation": 0.6},
        confidence="medium",
        models_used=["Manthana-ECG-Engine"],
        structures={"hr_bpm": 80.0, "signal_quality": "good"},
    )
    assert isinstance(r.structures, dict)
    assert r.structures["hr_bpm"] == 80.0


def test_correlation_ecg_xray_rules_registered():
    from correlation_engine import CORRELATION_RULES

    names = [r["name"] for r in CORRELATION_RULES]
    assert "ECG LVH with CXR Cardiomegaly" in names
    assert "ECG Atrial Fibrillation with CXR Cardiomegaly" in names
    assert "ECG LBBB with CXR Cardiomegaly" in names


def test_founder_merge_maps_afib_substring():
    from ecg_rhythm import RHYTHM_KEYS
    from founder_merge import merge_founder_into_rhythm_scores

    heuristic = {k: 0.15 for k in RHYTHM_KEYS}
    founder = {"paroxysmal_atrial_fibrillation": 0.9}
    out = merge_founder_into_rhythm_scores(founder, heuristic)
    assert out["atrial_fibrillation"] >= 0.85


def test_ecg_founder_infer_json_checkpoint(tmp_path, monkeypatch):
    import json
    import os

    p = tmp_path / "scores.json"
    p.write_text(
        json.dumps({"atrial_fibrillation": 0.88, "noise": "x"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ECG_FOUNDER_CHECKPOINT", str(p))
    monkeypatch.delenv("MODEL_DIR", raising=False)

    from ecg_founder_infer import predict_12lead

    import numpy as np

    sig = np.zeros((12, 500), dtype=np.float32)
    out = predict_12lead(sig, 500.0, str(tmp_path))
    assert out is not None
    assert out.get("atrial_fibrillation") == pytest.approx(0.88, abs=0.01)


def test_ecg_image_quality_flags_tiny_image(tmp_path):
    from PIL import Image

    from ecg_image_quality import assess_ecg_image_quality

    img_path = tmp_path / "tiny.jpg"
    Image.new("RGB", (32, 32), color="white").save(img_path, "JPEG")
    q = assess_ecg_image_quality(str(img_path), min_short_edge=480, blur_variance_min=15.0)
    assert q["ok"] is False
    assert q["warnings"]
