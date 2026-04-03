"""Tests for dermatology critical flags and V1 analyzer (mocked Kimi / OpenAI client)."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from io import BytesIO
from unittest.mock import MagicMock

import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DERM = os.path.join(_BACKEND, "services", "16_dermatology")
_SHARED = os.path.join(_BACKEND, "shared")


def _load_dermatology_analyzer():
    """Load 16_dermatology/analyzer.py without clashing with other services named analyzer."""
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    path = os.path.join(_DERM, "analyzer.py")
    spec = importlib.util.spec_from_file_location("manthana_dermatology_analyzer", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_melanoma_triggers_critical():
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    from critical_flags import check_derm_critical

    s = {c: 0.02 for c in [
        "tinea", "vitiligo", "psoriasis", "melasma", "acne",
        "eczema_dermatitis", "scabies", "urticaria", "bcc", "scc",
        "melanoma", "normal_benign",
    ]}
    s["melanoma"] = 0.72
    r = check_derm_critical(s)
    assert r["is_critical"] is True
    assert r["flag"] == "POSSIBLE_MALIGNANCY"
    assert r["top_malignancy"] == "melanoma"
    assert r["urgency"] == "URGENT"


def test_bcc_triggers_critical():
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    from critical_flags import check_derm_critical

    s = {c: 0.02 for c in [
        "tinea", "vitiligo", "psoriasis", "melasma", "acne",
        "eczema_dermatitis", "scabies", "urticaria", "scc", "melanoma", "normal_benign",
    ]}
    s["bcc"] = 0.55
    assert check_derm_critical(s)["is_critical"] is True


def test_combined_malignancy_triggers():
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    from critical_flags import check_derm_critical

    s = {c: 0.01 for c in [
        "tinea", "vitiligo", "psoriasis", "melasma", "acne",
        "eczema_dermatitis", "scabies", "urticaria", "normal_benign",
    ]}
    s["bcc"] = 0.18
    s["scc"] = 0.15
    s["melanoma"] = 0.14
    assert check_derm_critical(s)["is_critical"] is True


def test_tinea_not_critical():
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    from critical_flags import check_derm_critical

    s = {c: 0.02 for c in [
        "vitiligo", "psoriasis", "melasma", "acne",
        "eczema_dermatitis", "scabies", "urticaria", "bcc", "scc", "melanoma", "normal_benign",
    ]}
    s["tinea"] = 0.82
    r = check_derm_critical(s)
    assert r["is_critical"] is False
    assert r["urgency"] == "ROUTINE"


def test_parse_json_from_llm_embedded_in_prose():
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    az = _load_dermatology_analyzer()
    wrapped = 'Here is the result:\n```json\n{"tinea": 0.5, "vitiligo": 0.05, "psoriasis": 0.05, "melasma": 0.05, "acne": 0.05, "eczema_dermatitis": 0.05, "scabies": 0.05, "urticaria": 0.05, "bcc": 0.0, "scc": 0.0, "melanoma": 0.0, "normal_benign": 0.05, "top_class": "tinea", "confidence": 0.5, "confidence_label": "medium", "is_malignant_candidate": false}\n```\n'
    d = az._parse_json_from_llm(wrapped)
    assert d["top_class"] == "tinea"


def test_error_dict_graceful():
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    from critical_flags import check_derm_critical

    assert check_derm_critical({"error": "model_not_ready"})["is_critical"] is False


def _dummy_jpeg_b64():
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (10, 10), (180, 120, 80)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_analyze_v1_structure_mocked_kimi(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("KIMI_DERMATOLOGY_THINKING", "disabled")

    az = _load_dermatology_analyzer()

    az._derm_classifier = None

    scores_json = json.dumps({
        "tinea": 0.71, "vitiligo": 0.03, "psoriasis": 0.02,
        "melasma": 0.01, "acne": 0.01, "eczema_dermatitis": 0.02,
        "scabies": 0.05, "urticaria": 0.01, "bcc": 0.03,
        "scc": 0.02, "melanoma": 0.02, "normal_benign": 0.07,
        "top_class": "tinea", "confidence": 0.71,
        "confidence_label": "high", "is_malignant_candidate": False,
    })
    narrative_text = "4. IMPRESSION\nTinea corporis likely.\n\n### DIFFERENTIAL"

    mock_resp1 = MagicMock()
    mock_resp1.choices = [MagicMock(message=MagicMock(content=scores_json))]
    mock_resp2 = MagicMock()
    mock_resp2.choices = [MagicMock(message=MagicMock(content=narrative_text))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [mock_resp1, mock_resp2]

    monkeypatch.setattr(az, "_make_openai_client", lambda *a, **k: mock_client)

    result = az.analyze_dermatology(
        image_b64=_dummy_jpeg_b64(),
        patient_context={"age": 35, "sex": "M", "location_body": "trunk"},
        job_id="job-1",
    )

    assert result["modality"] == "dermatology"
    assert result["structures"]["classifier_mode"] == "kimi_k2.5_vision_v1"
    assert result["structures"]["critical"]["is_critical"] is False
    assert isinstance(result["findings"], list)
    assert len(result["findings"]) >= 1


def test_malignancy_critical_in_findings(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("KIMI_DERMATOLOGY_THINKING", "disabled")

    az = _load_dermatology_analyzer()

    az._derm_classifier = None

    malignant_scores = json.dumps({
        **{c: 0.01 for c in [
            "tinea", "vitiligo", "psoriasis", "melasma", "acne",
            "eczema_dermatitis", "scabies", "urticaria", "bcc", "normal_benign",
        ]},
        "scc": 0.65,
        "melanoma": 0.20,
        "top_class": "scc",
        "confidence": 0.65,
        "confidence_label": "high",
        "is_malignant_candidate": True,
    })
    narrative = "⚠️ URGENT DERMATOLOGY REFERRAL REQUIRED\nImpression: SCC suspected."

    mock_r1 = MagicMock()
    mock_r1.choices = [MagicMock(message=MagicMock(content=malignant_scores))]
    mock_r2 = MagicMock()
    mock_r2.choices = [MagicMock(message=MagicMock(content=narrative))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [mock_r1, mock_r2]

    monkeypatch.setattr(az, "_make_openai_client", lambda *a, **k: mock_client)

    result = az.analyze_dermatology(
        image_b64=_dummy_jpeg_b64(),
        patient_context={"age": 65, "sex": "M"},
        job_id="job-2",
    )

    assert result["structures"]["critical"]["is_critical"] is True
    assert result["structures"]["critical"]["urgency"] == "URGENT"
    assert any(
        f.get("severity") == "critical" for f in result["findings"]
    )
