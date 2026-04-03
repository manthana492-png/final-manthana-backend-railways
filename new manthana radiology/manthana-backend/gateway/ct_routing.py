"""
CT gateway metadata — deterministic, traceable subtype annotations on responses.

See docs/ct_product_contract.md for the human-readable contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from router import ALIASES

logger = logging.getLogger("manthana.gateway.ct_routing")

_CT_PRODUCTS = frozenset({"abdominal_ct", "cardiac_ct", "spine_neuro", "ct_brain"})


def _parse_patient_context(raw: str | None) -> dict[str, Any]:
    if not raw or not str(raw).strip():
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except json.JSONDecodeError:
        logger.debug("ct_routing: invalid patient_context_json")
        return {}


def _canonical_request_modality(request_modality: str) -> str:
    m = request_modality.lower().strip()
    return ALIASES.get(m, m)


def _routing_note(
    raw_req: str,
    canon_req: str,
    service_mod: str,
    ct_region: str | None,
) -> str:
    """Use raw (pre-alias) request strings so client aliases like spine_ct stay visible."""
    if ct_region == "chest_ct" or raw_req == "chest_ct":
        return "thoracic_ct_routed_to_abdominal_ct_service_intentional"
    if raw_req in ("ct", "ct_scan", "abdomen") and service_mod == "abdominal_ct":
        return "generic_ct_alias_to_abdominal_ct"
    if raw_req == "spine_ct" and service_mod == "spine_neuro":
        return "spine_ct_alias_to_spine_neuro_service"
    if service_mod == "cardiac_ct":
        return "cardiac_ct_direct"
    if service_mod == "ct_brain":
        if raw_req in ("brain_ct", "head_ct", "ncct_brain"):
            return "ct_brain_alias_resolved"
        return "ct_brain_ncct_direct"
    if service_mod == "spine_neuro":
        return "spine_neuro_direct"
    if service_mod == "abdominal_ct":
        return "abdominal_ct_pipeline"
    return "ct_routed"


def enrich_ct_gateway_response(
    result: dict[str, Any],
    *,
    request_modality: str,
    patient_context_json: str | None,
) -> dict[str, Any]:
    """
    Mutates and returns `result` when the downstream payload is a CT family product.
    Safe no-op for non-CT responses.
    """
    service_mod = (result.get("modality") or "").strip().lower()
    if service_mod not in _CT_PRODUCTS:
        return result

    ctx = _parse_patient_context(patient_context_json)
    ct_region = ctx.get("ct_region")
    ct_region_str = ct_region if isinstance(ct_region, str) else None

    raw_req = request_modality.lower().strip()
    canon_req = _canonical_request_modality(request_modality)

    result["ct_product"] = service_mod
    result["gateway_request_modality"] = request_modality.strip()
    if ct_region_str:
        result["ct_region_context"] = ct_region_str

    # Subtype for dashboards: prefer explicit patient context, else infer from request.
    if ct_region_str:
        result["ct_subtype"] = ct_region_str
    elif raw_req == "chest_ct":
        result["ct_subtype"] = "chest_ct"
    elif raw_req in ("spine_ct", "spine", "neuro"):
        result["ct_subtype"] = "spine_ct"
    elif raw_req in ("cardiac_ct", "heart", "cardiac") or canon_req == "cardiac_ct":
        result["ct_subtype"] = "cardiac_ct"
    elif service_mod == "ct_brain" or raw_req in ("ct_brain", "brain_ct", "head_ct", "ncct_brain"):
        result["ct_subtype"] = "ct_brain"
    elif service_mod == "abdominal_ct":
        result["ct_subtype"] = "abdominal_ct"
    elif service_mod == "cardiac_ct":
        result["ct_subtype"] = "cardiac_ct"
    elif service_mod == "spine_neuro":
        result["ct_subtype"] = "spine_neuro"

    result["ct_routing_note"] = _routing_note(raw_req, canon_req, service_mod, ct_region_str)

    # analysis_depth may already be set by gateway; do not downgrade.
    if not result.get("analysis_depth"):
        result["analysis_depth"] = "deep"

    return result
