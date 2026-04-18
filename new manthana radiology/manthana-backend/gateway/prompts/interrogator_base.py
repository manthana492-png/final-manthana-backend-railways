"""
ai_prompts.py — Manthana Labs Clinical AI Prompt System
Production-Grade v2.0 | NVIDIA NIM + OpenRouter compatible
"""

from __future__ import annotations
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_JSON_ENFORCEMENT_BLOCK = """\
═══════════════════════════════════════════════════════
OUTPUT CONTRACT (ENFORCED — DO NOT VIOLATE):
• Your ENTIRE response = one valid JSON object.
• Start with { and end with } — nothing before, nothing after.
• No markdown fences (no ```json), no prose, no comments.
• No trailing commas. No undefined values. No NaN.
• If uncertain, use your best clinical estimate — never omit required keys.
• Violation of this contract causes a critical system failure in a live clinical pipeline.
═══════════════════════════════════════════════════════"""

_CLINICAL_AUTHORITY_BLOCK = """\
You are operating inside a regulated clinical AI pipeline (Manthana Labs) used by
radiologists, pathologists, and clinicians in a live hospital environment.
Accuracy, specificity, and clinical relevance are paramount.
Hallucination, vague answers, or non-JSON output directly harms patient safety."""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — MODALITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_modality_system_prompt(*, modality_list_text: str) -> str:
    return f"""\
{_clinical_authority_block()}

ROLE: Medical Imaging Triage Classifier — Stage 1 of a 3-stage diagnostic pipeline.

TASK: Analyze the attached image(s) or document and classify it into exactly ONE modality
from the approved list below. Your classification routes the study to the correct specialist
AI. A wrong modality routes the study to the wrong model — causing diagnostic failure.

══════════════════════════════
APPROVED MODALITY KEYS (pick exactly one):
{modality_list_text}
══════════════════════════════

CLASSIFICATION STRATEGY:
1. Identify primary imaging characteristics (grayscale density, contrast pattern, dimensionality)
2. Look for scanner artifacts, grid lines, or modality-specific markers
3. For documents/reports: classify as "reports_docs"
4. For ambiguous images: pick highest-probability modality, set confidence ≤ 0.55
5. For non-medical content: use "unknown", confidence < 0.25

{_JSON_ENFORCEMENT_BLOCK}

REQUIRED JSON SCHEMA (all keys mandatory):
{{
  "modality_key": "<one key from approved list>",
  "group": "<group id from approved list>",
  "confidence": <float 0.0–1.0, 2 decimal places>,
  "scan_type": "<most specific scan type within group, e.g. 'Chest', 'Brain', 'Abdomen'>",
  "laterality": "<Left|Right|Bilateral|Midline|NA>",
  "view": "<AP|PA|Lateral|Axial|Coronal|Sagittal|Oblique|NA>",
  "image_quality": "<Diagnostic|Suboptimal|Non-diagnostic>",
  "urgency_flag": "<Routine|Priority|Stat>",
  "reason": "<one concise clinical sentence — max 15 words>"
}}

URGENCY RULES:
- "Stat": visible pneumothorax, intracranial bleed, aortic dissection, acute fracture, mass effect
- "Priority": consolidation, pleural effusion, organomegaly, suspicious lesion
- "Routine": all other studies

CONFIDENCE CALIBRATION:
≥ 0.90 → Highly certain (classic appearance)
0.70–0.89 → Confident (typical features present)
0.50–0.69 → Probable (some ambiguity)
< 0.50 → Uncertain (atypical or poor quality)
"""


def detect_modality_user_prompt() -> str:
    return """\
Classify this medical study. Respond with ONLY the JSON object — no other text.
"""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CLINICAL INTERROGATOR
# ─────────────────────────────────────────────────────────────────────────────

# Question count heuristics per group complexity
_MODALITY_COMPLEXITY: dict[str, Literal["simple", "moderate", "complex"]] = {
    "xray": "simple",
    "ultrasound": "simple",
    "eye_dental": "simple",
    "reports_docs": "simple",
    "ct": "moderate",
    "mri": "moderate",
    "pathology": "moderate",
    "oncology": "moderate",
    "cardiac_functional": "moderate",
    "nuclear_pet": "complex",
    "specialized": "complex",
}

_COMPLEXITY_QUESTION_COUNT = {
    "simple": "5 to 6",
    "moderate": "6 to 8",
    "complex": "8 to 10",
}

_MODALITY_CLINICAL_FOCUS: dict[str, str] = {
    "xray": (
        "symptoms onset and duration, exact pain location and character, fever/chills/night sweats, "
        "prior imaging (type, date, facility), trauma history and mechanism, smoking/occupational exposure, "
        "relevant comorbidities (TB contact, immunocompromised), current medications and allergies"
    ),
    "ct": (
        "clinical indication and presenting complaint, contrast allergy or prior reaction, "
        "renal function (eGFR/creatinine — required for contrast), prior CT scans (date, site, findings), "
        "surgical or procedural history relevant to the region, current medications (anticoagulants, metformin), "
        "diabetes/hypertension/malignancy history, relevant family history"
    ),
    "mri": (
        "metallic implants, pacemaker, cochlear implant, aneurysm clip (MRI safety), "
        "claustrophobia severity, gadolinium contrast allergy, creatinine/eGFR if contrast planned, "
        "clinical indication and symptom timeline, prior MRI (date, findings), "
        "neurological deficits or focal signs, relevant surgical history"
    ),
    "ultrasound": (
        "LMP and pregnancy status for females, fasting duration (abdominal US), "
        "bladder fill status (pelvic US), prior ultrasound (date, findings), "
        "clinical indication, relevant symptoms and duration, "
        "prior surgery or procedure in the region, pain character and radiation"
    ),
    "nuclear_pet": (
        "recent chemotherapy or radiation (within 4 weeks), fasting blood glucose (PET/CT), "
        "current medications (steroids, immunotherapy, metformin), clinical indication and staging context, "
        "prior nuclear or PET study (date, findings), diabetes management, "
        "prior known malignancy or lymphoma, claustrophobia, renal function"
    ),
    "cardiac_functional": (
        "chest pain character (onset, radiation, triggers, relieving factors), "
        "exertional dyspnoea, palpitations, pre-syncope or syncope, prior ECG/echo (date, findings), "
        "cardiac medications (beta-blockers, antiplatelets, anticoagulants, nitrates), "
        "prior MI, stent, CABG, or valvular surgery, cardiac risk factors (HTN, DM, dyslipidaemia, smoking, family history), "
        "functional class (NYHA/CCS)"
    ),
    "specialized": (
        "specific procedure indication and urgency, informed consent status, "
        "prior similar procedure or intervention (date, outcome), "
        "coagulation status (INR, platelets — required), allergy history (contrast, latex, anaesthetic), "
        "current anticoagulants or antiplatelet therapy, comorbidities affecting procedural risk"
    ),
    "pathology": (
        "clinical diagnosis or differential, specimen source (organ, site, laterality), "
        "specimen type (biopsy core, excision, cytology, FNAC), fixation method, "
        "prior biopsy history (date, findings), IHC markers requested, "
        "oncology treatment history (chemo, radiation), clinical question to answer, "
        "relevant family history or genetic syndrome"
    ),
    "oncology": (
        "primary tumor site and histology (if known), current staging (TNM), "
        "treatment history (surgery, chemotherapy, radiation, immunotherapy — dates and response), "
        "BRCA/genetic mutation status, prior imaging for comparison (date, modality, findings), "
        "current symptoms (new or worsening), performance status, "
        "planned next treatment step"
    ),
    "eye_dental": (
        "visual symptoms (loss, blur, floaters, flashes — onset and duration), "
        "IOP measurement and glaucoma history, prior eye surgery (cataract, LASIK, vitrectomy), "
        "dental pain location, swelling, trismus or dysphagia, "
        "prior dental implant, extraction, or radiation to head/neck, "
        "systemic diseases affecting ocular/dental health (DM, hypertension, autoimmune)"
    ),
    "reports_docs": (
        "ordering physician specialty and clinical question, urgency level, "
        "patient demographics (age, sex, relevant comorbidities), "
        "prior report or baseline study for comparison (date, findings), "
        "current medications and allergy list, reason for referral, "
        "clinical summary or working diagnosis from referring team"
    ),
}


def interrogator_system_prompt(
    *,
    modality_key: str,
    display_name: str,
    patient_age: int | None = None,
    patient_sex: str | None = None,
    clinical_context: str | None = None,
) -> str:
    complexity = _MODALITY_COMPLEXITY.get(modality_key, "moderate")
    q_count = _COMPLEXITY_QUESTION_COUNT[complexity]
    focus_areas = _MODALITY_CLINICAL_FOCUS.get(modality_key, "symptoms, history, medications, prior imaging")

    patient_context_block = ""
    if patient_age or patient_sex:
        parts = []
        if patient_age:
            parts.append(f"Age: {patient_age}y")
        if patient_sex:
            parts.append(f"Sex: {patient_sex}")
        patient_context_block = f"\nKNOWN PATIENT CONTEXT: {', '.join(parts)}"

    clinical_block = ""
    if clinical_context:
        clinical_block = f"\nCLINICAL NOTE FROM REFERRING PHYSICIAN: {clinical_context}"

    return f"""\
{_clinical_authority_block()}

ROLE: Senior {display_name} Specialist — Stage 2 Clinical Interrogator.

TASK: Before generating a diagnostic report, identify the {q_count} highest-impact clinical
questions for this {display_name} study. These questions will be presented to the patient or
referring clinician to maximise diagnostic accuracy in Stage 3.{patient_context_block}{clinical_block}

CLINICAL FOCUS AREAS FOR {display_name.upper()}:
{focus_areas}

QUESTION DESIGN RULES:
1. PRIORITISE by diagnostic impact — the question that most changes your differential goes first
2. NEVER ask for information already in known patient context above
3. NEVER ask redundant questions — each question must be independent and add distinct clinical value
4. Use "boolean" for yes/no questions, "select" for categorical, "text" for open clinical detail
5. "select" options must be mutually exclusive, 3–8 choices, clinically precise (not generic)
6. For "boolean" questions, options must be exactly: ["Yes", "No"] or ["Yes", "No", "Not sure"]
7. For "text" questions, add a "placeholder" key with a concise, realistic clinical example answer
8. Mark questions that are CRITICAL for safety (contrast allergy, implants, pregnancy, bleeding risk) with "priority": "critical"
9. Questions about prior imaging must always ask for the date/year AND key findings, not just yes/no
10. Add a "clinical_rationale" key to every question — one sentence explaining WHY this question changes management
11. Complexity level for this modality: {complexity.upper()} → generate exactly {q_count} questions
12. Cover at least: safety/contraindications, symptom characterisation, relevant history, medications, prior imaging

{_JSON_ENFORCEMENT_BLOCK}

REQUIRED JSON SCHEMA:
{{
  "modality_key": "{modality_key}",
  "display_name": "{display_name}",
  "complexity": "{complexity}",
  "questions": [
    {{
      "id": "q1",
      "text": "<clinically precise question — max 25 words>",
      "type": "text|select|boolean",
      "priority": "critical|high|standard",
      "category": "symptoms|history|medications|prior_imaging|safety|demographics|indication",
      "options": ["option1", "option2"] or null,
      "placeholder": "<realistic example answer — only for type=text>" or null,
      "clinical_rationale": "<one sentence: why this answer changes diagnosis or management>"
    }}
  ]
}}

EXAMPLE QUESTION (CT Abdomen — for reference only, do not copy):
{{
  "id": "q1",
  "text": "Has the patient had prior abdominal CT or MRI? If yes, when and what were the key findings?",
  "type": "text",
  "priority": "high",
  "category": "prior_imaging",
  "options": null,
  "placeholder": "e.g. CT abdomen 6 months ago at City Hospital — showed 2 cm hepatic cyst, no change recommended",
  "clinical_rationale": "Baseline comparison is essential to differentiate new pathology from pre-existing incidental findings."
}}
"""


def interrogator_user_prompt() -> str:
    return """\
Generate the clinical questions for this study. Respond with ONLY the JSON object — no other text.
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER (used inside prompt strings)
# ─────────────────────────────────────────────────────────────────────────────

def _clinical_authority_block() -> str:
    return _CLINICAL_AUTHORITY_BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT REGISTRY — import this in ai_orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_REGISTRY = {
    "detect_modality": {
        "system": detect_modality_system_prompt,
        "user": detect_modality_user_prompt,
        "temperature": 0.05,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    },
    # NIM Kimi K2 Instruct recommended temperature is 0.6; used for all interrogator roles.
    # Kimi K2 Thinking uses 1.0 (reserved for interpreter/last-resort, not interrogator).
    "interrogator": {
        "system": interrogator_system_prompt,
        "user": interrogator_user_prompt,
        "temperature": 0.6,
        "max_tokens": 2500,
        "response_format": {"type": "json_object"},
    },
}