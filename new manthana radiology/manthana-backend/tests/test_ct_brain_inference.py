from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
CT_BRAIN_SVC = BACKEND / "services" / "11_ct_brain"


@pytest.fixture
def ct_brain_inference(monkeypatch):
    monkeypatch.setenv("CT_BRAIN_CI_DUMMY_MODEL", "1")
    monkeypatch.setenv("CT_BRAIN_NARRATIVE_POLICY", "off")
    sys.path.insert(0, str(CT_BRAIN_SVC))
    for mod in ("inference",):
        sys.modules.pop(mod, None)
    import inference as inf

    yield importlib.reload(inf)


def test_ct_brain_pipeline_ci_dummy(ct_brain_inference, monkeypatch):
    inf = ct_brain_inference

    def fake_load(*_a, **_k):
        vol = np.zeros((32, 64, 64), dtype=np.float32)
        return vol, {"modality": "CT"}, True

    monkeypatch.setattr(inf, "load_ct_volume", fake_load)
    out = inf.run_pipeline(
        "/tmp/fake.dcm",
        "job-ctb-1",
        patient_context={"ct_brain_clinical_context": {"trauma_context": True}},
    )
    assert out["modality"] == "ct_brain"
    assert out["pathology_scores"]["inference_mode"] == "ci_dummy"
    assert "ich_probability" in out["pathology_scores"]
    assert "CT-Brain-CI-Dummy" in out["models_used"]
    assert out["structures"]["narrative_policy"] == "off"


def test_ct_brain_critical_finding_when_high_score(ct_brain_inference, monkeypatch):
    inf = ct_brain_inference
    monkeypatch.setenv("CT_BRAIN_CRITICAL_THRESHOLD", "0.05")

    def fake_load(*_a, **_k):
        return np.ones((8, 32, 32), dtype=np.float32), {"modality": "CT"}, True

    monkeypatch.setattr(inf, "load_ct_volume", fake_load)
    monkeypatch.setattr(inf, "_run_ci_dummy", lambda _t: 0.99)
    out = inf.run_pipeline("/x", "j2", patient_context=None)
    crit = [f for f in out["findings"] if f.get("severity") == "critical"]
    assert crit
    assert any("hemorrhage" in (f.get("label") or "").lower() for f in crit)


def test_ct_brain_narrative_policy_off_empty_narrative(ct_brain_inference, monkeypatch):
    inf = ct_brain_inference
    monkeypatch.setenv("CT_BRAIN_NARRATIVE_POLICY", "off")

    def fake_load(*_a, **_k):
        return np.zeros((8, 16, 16), dtype=np.float32), {"modality": "CT"}, True

    monkeypatch.setattr(inf, "load_ct_volume", fake_load)
    out = inf.run_pipeline("/x", "j3", patient_context=None)
    assert (out.get("structures") or {}).get("narrative_report") in ("", None)
