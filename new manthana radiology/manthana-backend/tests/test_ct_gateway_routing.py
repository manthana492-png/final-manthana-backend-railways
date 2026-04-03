from __future__ import annotations

from ct_routing import enrich_ct_gateway_response


def test_enrich_skips_non_ct_service():
    r = {"modality": "xray", "findings": []}
    out = enrich_ct_gateway_response(r, request_modality="xray", patient_context_json=None)
    assert "ct_product" not in out


def test_enrich_abdominal_ct_with_chest_context():
    r = {"modality": "abdominal_ct", "findings": []}
    ctx = '{"ct_region": "chest_ct"}'
    out = enrich_ct_gateway_response(
        r,
        request_modality="chest_ct",
        patient_context_json=ctx,
    )
    assert out["ct_product"] == "abdominal_ct"
    assert out["ct_region_context"] == "chest_ct"
    assert out["ct_subtype"] == "chest_ct"
    assert "thoracic" in out["ct_routing_note"]


def test_enrich_cardiac_direct():
    r = {"modality": "cardiac_ct"}
    out = enrich_ct_gateway_response(
        r,
        request_modality="cardiac_ct",
        patient_context_json="{}",
    )
    assert out["ct_product"] == "cardiac_ct"
    assert out["ct_subtype"] == "cardiac_ct"


def test_enrich_spine_neuro_alias():
    r = {"modality": "spine_neuro"}
    out = enrich_ct_gateway_response(
        r,
        request_modality="spine_ct",
        patient_context_json=None,
    )
    assert out["ct_product"] == "spine_neuro"
    assert out["ct_subtype"] == "spine_ct"


def test_enrich_ct_brain_direct():
    r = {"modality": "ct_brain", "findings": []}
    out = enrich_ct_gateway_response(
        r,
        request_modality="ct_brain",
        patient_context_json="{}",
    )
    assert out["ct_product"] == "ct_brain"
    assert out["ct_subtype"] == "ct_brain"
    assert "ct_brain" in out["ct_routing_note"]


def test_enrich_ct_brain_alias_head_ct():
    r = {"modality": "ct_brain", "findings": []}
    out = enrich_ct_gateway_response(
        r,
        request_modality="head_ct",
        patient_context_json=None,
    )
    assert out["ct_product"] == "ct_brain"
    assert out["ct_subtype"] == "ct_brain"
    assert "alias" in out["ct_routing_note"]
