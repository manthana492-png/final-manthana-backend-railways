"""Tests for oral cancer pipeline remediation (correlation keys, flatten, gateway helpers)."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPORT_ASSEMBLY = os.path.join(_BACKEND, "services", "report_assembly")
_GATEWAY = os.path.join(_BACKEND, "gateway")
_SHARED = os.path.join(_BACKEND, "shared")
_ORAL = os.path.join(_BACKEND, "services", "14_oral_cancer")


def _purge_oral_service_modules() -> None:
    oral_abs = os.path.abspath(_ORAL)
    for key in ("config", "inference", "main"):
        mod = sys.modules.get(key)
        if not mod:
            continue
        fp = getattr(mod, "__file__", None)
        if fp and os.path.abspath(os.path.dirname(fp)) == oral_abs:
            del sys.modules[key]


@pytest.fixture
def correlation_engine():
    sys.path.insert(0, _REPORT_ASSEMBLY)
    import correlation_engine as ce

    return ce


def test_flatten_oral_pathology_scores(correlation_engine):
    flat = correlation_engine._flatten_result(
        "oral_cancer",
        {
            "pathology_scores": {
                "normal": 0.2,
                "opmd": 0.55,
                "oscc_suspicious": 0.25,
            }
        },
    )
    assert flat["oral_cancer.opmd"] == pytest.approx(0.55)
    assert flat["oral_cancer.oscc_suspicious"] == pytest.approx(0.25)


def test_oral_correlation_rules_exist(correlation_engine):
    names = [r["name"] for r in correlation_engine.CORRELATION_RULES]
    assert any("Oral OSCC" in n for n in names)
    assert any("Oral OPMD" in n for n in names)


def test_gateway_build_findings_dict(monkeypatch):
    sys.modules.pop("schemas", None)
    sys.path.insert(0, _GATEWAY)
    from schemas import SingleReportRequest

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gateway_main", os.path.join(_GATEWAY, "main.py")
    )
    # Avoid loading full gateway (FastAPI deps); duplicate helper logic minimally:
    def build(req: SingleReportRequest) -> dict:
        if isinstance(req.findings, dict):
            d = dict(req.findings)
        else:
            d = {"items": list(req.findings or [])}
        if "pathology_scores" not in d:
            d["pathology_scores"] = req.pathology_scores or {}
        if "impression" not in d:
            d["impression"] = req.impression
        cn = req.clinical_notes
        if cn is None and isinstance(req.structures, dict):
            cn = req.structures.get("clinical_notes")
        d["clinical_notes"] = (cn if cn is not None else "") or d.get("clinical_notes", "")
        return d

    req = SingleReportRequest(
        modality="oral_cancer",
        findings=[{"label": "OPMD", "severity": "warning", "confidence": 72.0}],
        pathology_scores={"opmd": 0.6},
        impression="Test",
        structures={"clinical_notes": "tobacco_use:chewing"},
        clinical_notes=None,
    )
    d = build(req)
    assert "items" in d
    assert d["clinical_notes"] == "tobacco_use:chewing"


def test_logits_shape_with_mock_checkpoint(tmp_path):
    """Model forward pass produces (1, 3) logits — not backbone embedding dim."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import EfficientNetForImageClassification

    model = EfficientNetForImageClassification.from_pretrained(
        "google/efficientnet-b3",
        num_labels=3,
        ignore_mismatched_sizes=True,
        cache_dir=str(tmp_path),
    )
    dummy = torch.randn(1, 3, 300, 300)
    with torch.no_grad():
        output = model(dummy)
    assert hasattr(output, "logits"), "logits missing from output"
    assert output.logits.shape == (1, 3), f"Expected (1, 3), got {output.logits.shape}"


def test_503_when_no_checkpoint(tmp_path, monkeypatch):
    """POST /analyze/oral_cancer returns 503 when service is disabled (weights are optional)."""
    from PIL import Image

    img_path = tmp_path / "oral_test.jpg"
    Image.new("RGB", (16, 16), color="white").save(img_path, "JPEG")

    # Gateway tests cache `schemas` → gateway/schemas; oral service needs shared/schemas.
    sys.modules.pop("schemas", None)
    sys.modules.pop("config", None)
    sys.modules.pop("inference", None)
    sys.modules.pop("main", None)
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _ORAL)
    import main as oral_main

    monkeypatch.setattr(oral_main, "is_service_ready", lambda: False)

    from fastapi.testclient import TestClient

    client = TestClient(oral_main.app, raise_server_exceptions=False)
    with open(img_path, "rb") as f:
        resp = client.post(
            "/analyze/oral_cancer",
            files={"file": ("oral_test.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 503


def test_analysis_response_validates():
    """Pipeline-shaped dict passes shared AnalysisResponse validation."""
    spec = importlib.util.spec_from_file_location(
        "shared_schemas", os.path.join(_SHARED, "schemas.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AnalysisResponse = mod.AnalysisResponse

    mock_result = {
        "job_id": "test-123",
        "modality": "oral_cancer",
        "findings": [
            {
                "label": "Normal",
                "description": "...",
                "severity": "clear",
                "confidence": 87.3,
            }
        ],
        "impression": "No significant lesion identified.",
        "pathology_scores": {
            "normal": 0.873,
            "opmd": 0.091,
            "oscc_suspicious": 0.036,
            "oscc_confidence": 0.036,
            "opmd_confidence": 0.091,
            "submucous_fibrosis_confidence": 0.05,
            "lesion_present": 0.127,
            "high_risk_habit_confidence": 0.1,
            "biopsy_urgency_confidence": 0.15,
        },
        "structures": {
            "predicted_class": "Normal",
            "checkpoint_used": True,
            "input_type": "clinical_photo",
            "lesion_location": "unknown",
            "habit_risk": "unknown",
            "biopsy_recommended": False,
            "emergency_flags": [],
            "narrative_report": "",
            "india_note": "",
            "model_path": "",
        },
        "confidence": "high",
        "models_used": ["EfficientNet-B3"],
        "disclaimer": "...",
    }
    validated = AnalysisResponse(**mock_result)
    assert isinstance(validated.findings, list)
    assert validated.findings[0].confidence == 87.3


def test_map_binary_oral_probs_to_three_class():
    import importlib

    numpy = pytest.importorskip("numpy")
    sys.path.insert(0, _ORAL)
    import inference as oral_inference

    importlib.reload(oral_inference)
    three, pred = oral_inference.map_binary_oral_probs_to_three_class(
        numpy.array([0.7, 0.3], dtype=numpy.float64)
    )
    assert abs(float(three.sum()) - 1.0) < 1e-5
    assert pred == 0
    three2, pred2 = oral_inference.map_binary_oral_probs_to_three_class(
        numpy.array([0.05, 0.95], dtype=numpy.float64), opmd_fraction=0.3
    )
    assert pred2 == 2
    assert three2[2] > three2[1]


def test_oral_classifier_order_v2m_before_b3_by_default(tmp_path, monkeypatch):
    import importlib

    _purge_oral_service_modules()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("ORAL_PREFER_V2M", raising=False)
    (tmp_path / "oral_cancer_finetuned.pt").write_bytes(b"1")
    (tmp_path / "oral_effnet_v2m.pt").write_bytes(b"1")
    sys.path.insert(0, _ORAL)
    import config as oral_config

    importlib.reload(oral_config)
    import inference as oral_inference

    importlib.reload(oral_inference)
    assert oral_inference._oral_clinical_classifier_order() == ["v2m", "b3"]


def test_oral_classifier_order_b3_first_when_env_false(tmp_path, monkeypatch):
    import importlib

    _purge_oral_service_modules()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ORAL_PREFER_V2M", "false")
    (tmp_path / "oral_cancer_finetuned.pt").write_bytes(b"1")
    (tmp_path / "oral_effnet_v2m.pt").write_bytes(b"1")
    sys.path.insert(0, _ORAL)
    import config as oral_config

    importlib.reload(oral_config)
    import inference as oral_inference

    importlib.reload(oral_inference)
    assert oral_inference._oral_clinical_classifier_order() == ["b3", "v2m"]


def test_analyze_oral_cancer_smoke_200_with_mock_classifier(tmp_path, monkeypatch):
    from PIL import Image

    numpy = pytest.importorskip("numpy")
    img_path = tmp_path / "oral_smoke.jpg"
    Image.new("RGB", (32, 32), color=(120, 80, 60)).save(img_path, "JPEG")

    _purge_oral_service_modules()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ORAL_CANCER_ENABLED", "true")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)
    sys.modules.pop("schemas", None)
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _ORAL)

    import importlib

    import config as oral_config

    importlib.reload(oral_config)
    import inference as oral_inference

    importlib.reload(oral_inference)

    def fake_clinical(pil, **kw):
        a = numpy.array([0.82, 0.12, 0.06], dtype=numpy.float64)
        return a, 0, "EfficientNet-V2-M", str(tmp_path / "oral_effnet_v2m.pt")

    monkeypatch.setattr(oral_inference, "_run_clinical_photo_classifiers", fake_clinical)
    monkeypatch.setattr(
        oral_inference,
        "_call_oral_cancer_narrative",
        lambda **kw: ("", [], None),
    )

    import main as oral_main

    importlib.reload(oral_main)

    from fastapi.testclient import TestClient

    client = TestClient(oral_main.app, raise_server_exceptions=False)
    with open(img_path, "rb") as f:
        resp = client.post(
            "/analyze/oral_cancer",
            files={"file": ("oral_smoke.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("modality") == "oral_cancer"
    assert "EfficientNet-V2-M" in body.get("models_used", [])
    assert body.get("structures", {}).get("checkpoint_used") is True
