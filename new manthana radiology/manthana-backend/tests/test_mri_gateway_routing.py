from __future__ import annotations

from mri_routing import enrich_mri_gateway_response
from router import route_to_service


def test_route_spine_mri_alias_to_spine_neuro():
    url = route_to_service("spine_mri")
    assert "spine_neuro" in url


def test_route_mr_spine_alias():
    url = route_to_service("mr_spine")
    assert "spine_neuro" in url


def test_enrich_mri_skips_non_brain_service():
    r = {"modality": "xray", "findings": []}
    out = enrich_mri_gateway_response(r, request_modality="mri")
    assert "mri_product" not in out


def test_enrich_mri_brain_mri_from_mri_alias():
    r = {"modality": "brain_mri", "findings": []}
    out = enrich_mri_gateway_response(r, request_modality="mri")
    assert out["mri_product"] == "brain_mri"
    assert out["gateway_request_modality"] == "mri"
    assert "alias" in out["mri_routing_note"]


def test_enrich_mri_brain_mri_direct():
    r = {"modality": "brain_mri", "findings": []}
    out = enrich_mri_gateway_response(r, request_modality="brain_mri")
    assert out["mri_product"] == "brain_mri"
    assert out["mri_routing_note"] == "brain_mri_direct"


def test_zeroclaw_analyze_brain_mri_registered():
    from zeroclaw_tools import TOOL_EXECUTORS

    assert callable(TOOL_EXECUTORS.get("analyze_brain_mri"))
