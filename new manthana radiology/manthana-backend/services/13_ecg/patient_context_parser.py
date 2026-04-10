"""
ECG patient_context_json — normalize nested `patient_context` and format for LLM prompts.
Aligns with frontend ECG intake (India prompt-engineering guide schema).
"""

from __future__ import annotations

import json
from typing import Any, Optional

QT_SUBSTRINGS = (
    "azithromycin",
    "clarithromycin",
    "erythromycin",
    "ciprofloxacin",
    "levofloxacin",
    "moxifloxacin",
    "fluoroquinolone",
    "hydroxychloroquine",
    "chloroquine",
    "amiodarone",
    "sotalol",
    "haloperidol",
    "lithium",
    "tricyclic",
    "amitriptyline",
    "fluconazole",
    "itraconazole",
    "ondansetron",
    "domperidone",
    "digoxin",
    "methadone",
)


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def nested_patient_context(top: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return inner `patient_context` object if present and dict, else {}."""
    if not top or not isinstance(top, dict):
        return {}
    inner = top.get("patient_context")
    if isinstance(inner, dict):
        return inner
    return {}


def age_risk_note(age: Any, sex: str) -> str:
    a = _as_int(age)
    sx = (sex or "").strip().upper()
    if a is None:
        return "Age not fully specified — use conservative thresholds and clinical correlation."
    if a < 30:
        return (
            "Young patient — prioritize channelopathy, HCM, myocarditis, rheumatic heart disease "
            "in Indian context; premature CAD less likely but not impossible."
        )
    if a < 45 and sx == "M":
        return (
            "PREMATURE CAD RISK — Indian males often present with MI a decade earlier than Western "
            "norms; interpret ST/T and reciprocal changes with high suspicion when clinically appropriate."
        )
    if a < 45:
        return (
            "Young/mid adult — consider rheumatic valvular disease, cardiomyopathy, and autoimmune "
            "cardiac involvement alongside ischemia."
        )
    if a < 60:
        return (
            "Peak atherosclerotic CAD age band for many Indian patients — integrate hypertension, "
            "diabetes, and metabolic risk in interpretation."
        )
    if a < 75:
        return "Older adult — multi-morbidity, polypharmacy (QT/digoxin), and conduction disease more common."
    return "Elderly — sick sinus, complete heart block, severe valve disease, and drug-induced ECG changes."


def medication_qt_flags(medications: Any) -> str:
    if not isinstance(medications, list):
        return "None identified from structured medication list."
    flagged: list[str] = []
    for item in medications:
        if not isinstance(item, dict):
            continue
        drug = str(item.get("drug", "")).lower().strip()
        if not drug:
            continue
        if any(s in drug for s in QT_SUBSTRINGS):
            flagged.append(str(item.get("drug", "")).strip())
    if not flagged:
        return "None identified from structured medication list."
    return "QT / repolarization risk medications: " + ", ".join(flagged)


def family_genetic_flags(inner: dict[str, Any]) -> str:
    fh = inner.get("family_history")
    if not isinstance(fh, dict):
        return "No structured family history supplied."
    parts: list[str] = []
    if fh.get("premature_cad"):
        parts.append("Premature CAD in family — consider elevated Lp(a), familial hypercholesterolemia.")
    if fh.get("hcm_or_cardiomyopathy"):
        parts.append("HCM/cardiomyopathy family history — MYBPC3 and other South Asian founder variants.")
    if fh.get("long_qt_or_channelopathy"):
        parts.append("Channelopathy family history — scrutinize QTc and Brugada morphology.")
    if fh.get("sudden_cardiac_death"):
        parts.append("Sudden cardiac death in family — arrhythmic risk.")
    if fh.get("familial_hypercholesterolemia"):
        parts.append("FH phenotype possible — premature atherosclerosis.")
    return "; ".join(parts) if parts else "No major genetic flags documented in form."


def india_context_bullets(inner: dict[str, Any]) -> list[str]:
    """Short India-specific reminders from structured toggles."""
    out: list[str] = []
    mh = inner.get("medical_history")
    if isinstance(mh, dict):
        if mh.get("tuberculosis_history"):
            out.append("TB history: consider pericarditis/conduction involvement if diffuse ECG changes + symptoms.")
        if mh.get("copd_asthma"):
            out.append("COPD/asthma: RV strain, P pulmonale, S1Q3T3 mimic — correlate clinically.")
        if mh.get("diabetes"):
            out.append("Diabetes: silent ischemia possible; do not dismiss atypical presentations.")
    ch = inner.get("cardiac_history")
    if isinstance(ch, dict) and ch.get("rheumatic_fever_history"):
        out.append("Rheumatic fever history: MS/MR and AF in younger patients — echo correlation when indicated.")
    demo = inner.get("demographics")
    if isinstance(demo, dict):
        region = str(demo.get("state_region", "") or "").lower()
        if any(x in region for x in ("kerala", "tamil", "karnataka", "andhra", "telangana")):
            out.append("South India context: Brugada / channelopathy awareness in appropriate patterns.")
    return out


def format_ecg_patient_prompt_section(patient_context: Optional[dict[str, Any]]) -> str:
    """
    Build a single text block appended to the user message for narrative_ecg.
    `patient_context` is the top-level dict from patient_context_json (may include nested patient_context).
    """
    if not patient_context or not isinstance(patient_context, dict):
        return ""

    inner = nested_patient_context(patient_context)
    age = patient_context.get("age")
    if age is None and isinstance(inner.get("demographics"), dict):
        age = inner["demographics"].get("age")
    sex = str(patient_context.get("sex", "") or "")
    if not sex and isinstance(inner.get("demographics"), dict):
        dsex = inner["demographics"].get("sex")
        if dsex:
            sex = str(dsex)

    lines: list[str] = [
        "### PATIENT_CONTEXT (structured JSON — use for personalization)",
    ]
    if inner:
        try:
            lines.append(json.dumps(inner, indent=2, ensure_ascii=False))
        except (TypeError, ValueError):
            lines.append(str(inner))
    else:
        lines.append("{}")

    lines.append("")
    lines.append("### COMPUTED CLINICAL STRATIFICATION (for narrative — do not ignore)")
    lines.append(f"- Age–risk note: {age_risk_note(age, sex)}")
    lines.append(f"- Medication QT flags: {medication_qt_flags(inner.get('current_medications'))}")
    lines.append(f"- Family / genetic hints: {family_genetic_flags(inner)}")
    bullets = india_context_bullets(inner)
    if bullets:
        lines.append("- India-context triggers from form:")
        for b in bullets:
            lines.append(f"  • {b}")

    ch = str(patient_context.get("clinical_history", "") or "").strip()
    sym = str(patient_context.get("symptoms", "") or "").strip()
    if ch:
        lines.append("")
        lines.append("### ADDITIONAL free-text clinical_history (clinician)")
        lines.append(ch)
    if sym:
        lines.append("")
        lines.append("### ADDITIONAL free-text symptoms")
        lines.append(sym)

    return "\n".join(lines).strip()
