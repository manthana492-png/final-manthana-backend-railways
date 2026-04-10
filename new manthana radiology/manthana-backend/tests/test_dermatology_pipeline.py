"""Tests for dermatology critical flags and V1 analyzer (mocked OpenRouter / llm_router)."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from io import BytesIO
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


def test_analyze_v1_structure_mocked_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-xxxxxxxx")
    monkeypatch.setenv("DERM_CLASSIFIER_PRIORITY", "openrouter")

    az = _load_dermatology_analyzer()

    scores_json = json.dumps({
        "tinea": 0.71, "vitiligo": 0.03, "psoriasis": 0.02,
        "melasma": 0.01, "acne": 0.01, "eczema_dermatitis": 0.02,
        "scabies": 0.05, "urticaria": 0.01, "bcc": 0.03,
        "scc": 0.02, "melanoma": 0.02, "normal_benign": 0.07,
        "top_class": "tinea", "confidence": 0.71,
        "confidence_label": "high", "is_malignant_candidate": False,
    })
    narrative_text = "4. IMPRESSION\nTinea corporis likely.\n\n### DIFFERENTIAL"

    def _fake_openrouter(**kwargs):
        if kwargs.get("requires_json"):
            return scores_json, "openai/gpt-4o-mini"
        return narrative_text, "openai/gpt-4o-mini"

    monkeypatch.setattr(az, "_openrouter_derm_complete", _fake_openrouter)

    result = az.analyze_dermatology(
        image_b64=_dummy_jpeg_b64(),
        patient_context={"age": 35, "sex": "M", "location_body": "trunk"},
        job_id="job-1",
    )

    assert result["modality"] == "dermatology"
    assert result["structures"]["classifier_mode"] == "openrouter_vision_v1"
    assert result["structures"]["critical"]["is_critical"] is False
    assert isinstance(result["findings"], list)
    assert len(result["findings"]) >= 1


def test_malignancy_critical_in_findings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-xxxxxxxx")
    monkeypatch.setenv("DERM_CLASSIFIER_PRIORITY", "openrouter")

    az = _load_dermatology_analyzer()

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

    def _fake_openrouter(**kwargs):
        if kwargs.get("requires_json"):
            return malignant_scores, "openai/gpt-4o-mini"
        return narrative, "openai/gpt-4o-mini"

    monkeypatch.setattr(az, "_openrouter_derm_complete", _fake_openrouter)

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


def test_ham_hint_triggers_critical_when_ham_mass_high():
    sys.path.insert(0, _SHARED)
    sys.path.insert(0, _DERM)
    from classifier import DERM_CLASSES
    from critical_flags import check_derm_critical

    s = {c: 0.05 for c in DERM_CLASSES}
    s["normal_benign"] = 0.4
    hint = {
        "ham_mel": 0.5,
        "ham_bcc": 0.0,
        "ham_akiec": 0.0,
        "ham_combined_malignancy": 0.5,
    }
    r = check_derm_critical(s, ham_hint=hint)
    assert r["is_critical"] is True
    assert r.get("ham_malignancy_hint") == hint


def test_ham7_mapping_normalizes_to_derm_classes():
    sys.path.insert(0, _DERM)
    from classifier import DERM_CLASSES
    from ham_map import ham7_probs_to_derm_scores

    order = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    probs = {k: 1.0 / 7.0 for k in order}
    raw7, full = ham7_probs_to_derm_scores(probs, order)
    assert len(raw7) == 7
    total = sum(float(full[k]) for k in DERM_CLASSES)
    assert abs(total - 1.0) < 0.02
    assert full["top_class"] in DERM_CLASSES


def test_b4_branch_skips_openrouter_scores(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-xxxxxxxx")
    monkeypatch.setenv("DERM_CLASSIFIER_PRIORITY", "b4,openrouter")

    az = _load_dermatology_analyzer()
    sys.path.insert(0, _DERM)
    from classifier import DERM_CLASSES

    class FB4:
        def model_for_cam(self):
            return None

        transform = None

        def classify(self, pil):
            d = {c: round(1.0 / 12, 4) for c in DERM_CLASSES}
            d.update(
                {
                    "top_class": "tinea",
                    "confidence": 0.12,
                    "confidence_label": "low",
                    "is_malignant_candidate": False,
                }
            )
            return d

    monkeypatch.setattr(az, "_try_load_ham_classifier", lambda: None)
    monkeypatch.setattr(az, "_try_load_b4_classifier", lambda: FB4())

    calls = {"narrative": 0, "scores": 0}

    def _fake_openrouter(**kwargs):
        if kwargs.get("requires_json"):
            calls["scores"] += 1
            return "{}", "bad"
        calls["narrative"] += 1
        return "IMPRESSION\nTest OK.\n", "openai/gpt-4o-mini"

    monkeypatch.setattr(az, "_openrouter_derm_complete", _fake_openrouter)

    result = az.analyze_dermatology(
        image_b64=_dummy_jpeg_b64(),
        patient_context={},
        job_id="job-b4",
    )
    assert calls["scores"] == 0
    assert calls["narrative"] == 1
    assert result["structures"]["classifier_mode"] == "efficientnet_b4"
    assert "EfficientNet-B4-derm" in result["models_used"]


def test_analysisresponse_model_accepts_openrouter_pipeline(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-xxxxxxxx")
    monkeypatch.setenv("DERM_CLASSIFIER_PRIORITY", "openrouter")

    az = _load_dermatology_analyzer()
    scores_json = json.dumps({
        "tinea": 0.71,
        "vitiligo": 0.03,
        "psoriasis": 0.02,
        "melasma": 0.01,
        "acne": 0.01,
        "eczema_dermatitis": 0.02,
        "scabies": 0.05,
        "urticaria": 0.01,
        "bcc": 0.03,
        "scc": 0.02,
        "melanoma": 0.02,
        "normal_benign": 0.07,
        "top_class": "tinea",
        "confidence": 0.71,
        "confidence_label": "high",
        "is_malignant_candidate": False,
    })

    def _fake_openrouter(**kwargs):
        if kwargs.get("requires_json"):
            return scores_json, "openai/gpt-4o-mini"
        return "IMPRESSION\nOK.\n", "openai/gpt-4o-mini"

    monkeypatch.setattr(az, "_openrouter_derm_complete", _fake_openrouter)
    result = az.analyze_dermatology(
        image_b64=_dummy_jpeg_b64(),
        patient_context={},
        job_id="job-ar",
    )
    result["processing_time_sec"] = 0.1
    sys.path.insert(0, _SHARED)
    from schemas import AnalysisResponse

    AnalysisResponse(**result)
