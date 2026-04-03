"""Urgency detection from condition probability scores (V1 Claude or V2 classifier)."""

from __future__ import annotations

from typing import Any

MALIGNANCY_THRESHOLD = 0.50
COMBINED_THRESHOLD = 0.40


def check_derm_critical(condition_scores: dict[str, Any]) -> dict[str, Any]:
    if "error" in condition_scores:
        return {"is_critical": False, "flag": None, "urgency": "ROUTINE"}

    bcc = float(condition_scores.get("bcc", 0.0) or 0.0)
    scc = float(condition_scores.get("scc", 0.0) or 0.0)
    mel = float(condition_scores.get("melanoma", 0.0) or 0.0)
    combined = bcc + scc + mel

    malignant = {"bcc": bcc, "scc": scc, "melanoma": mel}
    top = max(malignant, key=malignant.get)
    top_score = malignant[top]
    is_critical = top_score >= MALIGNANCY_THRESHOLD or combined >= COMBINED_THRESHOLD

    if is_critical:
        return {
            "is_critical": True,
            "flag": "POSSIBLE_MALIGNANCY",
            "top_malignancy": top,
            "top_malignancy_score": round(top_score, 4),
            "combined_malignancy_score": round(combined, 4),
            "urgency": "URGENT",
            "action": "Refer to dermatologist or oncologist — do not delay",
        }
    return {"is_critical": False, "flag": None, "urgency": "ROUTINE"}
