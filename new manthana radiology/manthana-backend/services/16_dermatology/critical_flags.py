"""Urgency detection from condition probability scores (V1 Claude or V2 classifier)."""

from __future__ import annotations

from typing import Any

MALIGNANCY_THRESHOLD = 0.50
COMBINED_THRESHOLD = 0.40
# HAM7 raw mass (mel + bcc + 0.5*akiec) — see ham_map.ham_malignancy_hint
HAM_COMBINED_CRITICAL = 0.42


def check_derm_critical(
    condition_scores: dict[str, Any],
    ham_hint: dict[str, float] | None = None,
) -> dict[str, Any]:
    if "error" in condition_scores:
        return {"is_critical": False, "flag": None, "urgency": "ROUTINE"}

    bcc = float(condition_scores.get("bcc", 0.0) or 0.0)
    scc = float(condition_scores.get("scc", 0.0) or 0.0)
    mel = float(condition_scores.get("melanoma", 0.0) or 0.0)
    derm_combined = bcc + scc + mel

    ham_mass = 0.0
    if ham_hint:
        ham_mass = float(ham_hint.get("ham_combined_malignancy", 0.0) or 0.0)

    malignant = {"bcc": bcc, "scc": scc, "melanoma": mel}
    top = max(malignant, key=malignant.get)
    top_score = malignant[top]
    is_critical = (
        top_score >= MALIGNANCY_THRESHOLD
        or derm_combined >= COMBINED_THRESHOLD
        or (ham_hint is not None and ham_mass >= HAM_COMBINED_CRITICAL)
    )

    combined_report = max(derm_combined, ham_mass) if ham_hint else derm_combined

    if is_critical:
        display_top_score = max(top_score, ham_mass)
        return {
            "is_critical": True,
            "flag": "POSSIBLE_MALIGNANCY",
            "top_malignancy": top,
            "top_malignancy_score": round(float(display_top_score), 4),
            "combined_malignancy_score": round(combined_report, 4),
            "urgency": "URGENT",
            "action": "Refer to dermatologist or oncologist — do not delay",
            "ham_malignancy_hint": ham_hint,
        }
    out: dict[str, Any] = {"is_critical": False, "flag": None, "urgency": "ROUTINE"}
    if ham_hint:
        out["ham_malignancy_hint"] = ham_hint
    return out
