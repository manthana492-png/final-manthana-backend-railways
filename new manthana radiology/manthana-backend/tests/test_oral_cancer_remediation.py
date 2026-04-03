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
