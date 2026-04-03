"""
MRI gateway metadata — deterministic routing annotations on brain_mri responses.

See docs/mri_product_contract.md.
"""

from __future__ import annotations

from typing import Any

from router import ALIASES


def _canonical_request_modality(request_modality: str) -> str:
    m = request_modality.lower().strip()
    return ALIASES.get(m, m)


def enrich_mri_gateway_response(
    result: dict[str, Any],
    *,
    request_modality: str,
) -> dict[str, Any]:
    """
    Mutates and returns `result` when downstream payload is brain_mri.
    Safe no-op for other modalities.
    """
    service_mod = (result.get("modality") or "").strip().lower()
    if service_mod != "brain_mri":
        return result

    raw_req = request_modality.strip()
    canon_req = _canonical_request_modality(request_modality)

    result["mri_product"] = "brain_mri"
    result["gateway_request_modality"] = raw_req

    if raw_req.lower() in ("mri", "brain", "head_mri"):
        result["mri_routing_note"] = f"alias_{raw_req.lower()}_to_brain_mri"
    elif canon_req == "brain_mri":
        result["mri_routing_note"] = "brain_mri_direct"
    else:
        result["mri_routing_note"] = "brain_mri_service"

    if not result.get("analysis_depth"):
        result["analysis_depth"] = "deep"

    return result
