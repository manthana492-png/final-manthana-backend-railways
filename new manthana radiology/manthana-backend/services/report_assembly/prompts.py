"""
Manthana — Report Prompt Templates
Build modality-specific prompts for the LLM.
Supports 10 Indian languages with native-script section headers.
"""

import json
from typing import Optional


# ══════════════════════════════════════════════════════════
# Language Configuration
# 10 Indian languages + English
# ══════════════════════════════════════════════════════════

LANGUAGE_CONFIGS = {
    "en": {
        "name": "English",
        "script": "Latin",
        "findings_header": "FINDINGS",
        "impression_header": "IMPRESSION",
        "instruction": "Write in clear, professional medical English.",
        "disclaimer": "This is an AI-generated second opinion, not a primary diagnosis.",
        "direction": "ltr",
    },
    "hi": {
        "name": "Hindi",
        "script": "Devanagari",
        "findings_header": "निष्कर्ष",
        "impression_header": "निदान",
        "instruction": (
            "पूरी रिपोर्ट हिंदी में लिखें। "
            "तकनीकी शब्द देवनागरी में लिखें। "
            "चिकित्सा शब्दावली को सरल हिंदी में समझायें। "
            "FINDINGS और IMPRESSION की जगह हिंदी शीर्षक उपयोग करें।"
        ),
        "disclaimer": "यह एक AI-जनित द्वितीय राय है, प्राथमिक निदान नहीं।",
        "direction": "ltr",
    },
    "ta": {
        "name": "Tamil",
        "script": "Tamil",
        "findings_header": "கண்டுபிடிப்புகள்",
        "impression_header": "முடிவு",
        "instruction": (
            "முழு அறிக்கையையும் தமிழில் எழுதுங்கள். "
            "மருத்துவ சொற்களை தமிழ் வழக்கில் எழுதவும். "
            "FINDINGS இன் பதிலாக 'கண்டுபிடிப்புகள்:' மற்றும் "
            "IMPRESSION இன் பதிலாக 'முடிவு:' என்று தலைப்பு வையுங்கள்."
        ),
        "disclaimer": "இது AI உருவாக்கிய இரண்டாவது கருத்து, முதன்மை நோயறிதல் அல்ல.",
        "direction": "ltr",
    },
    "te": {
        "name": "Telugu",
        "script": "Telugu",
        "findings_header": "నిర్ధారణలు",
        "impression_header": "అభిప్రాయం",
        "instruction": (
            "మొత్తం నివేదికను తెలుగులో రాయండి. "
            "వైద్య పరిభాషను సరళమైన తెలుగులో వివరించండి. "
            "FINDINGS బదులు 'నిర్ధారణలు:' మరియు "
            "IMPRESSION బదులు 'అభిప్రాయం:' అనే శీర్షికలు వాడండి."
        ),
        "disclaimer": "ఇది AI రూపొందించిన రెండవ అభిప్రాయం, ప్రాథమిక నిర్ధారణ కాదు.",
        "direction": "ltr",
    },
    "kn": {
        "name": "Kannada",
        "script": "Kannada",
        "findings_header": "ಸಂಶೋಧನೆಗಳು",
        "impression_header": "ಅಭಿಪ್ರಾಯ",
        "instruction": (
            "ಸಂಪೂರ್ಣ ವರದಿಯನ್ನು ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ. "
            "ವೈದ್ಯಕೀಯ ಪದಗಳನ್ನು ಸರಳ ಕನ್ನಡದಲ್ಲಿ ವಿವರಿಸಿ. "
            "FINDINGS ಬದಲು 'ಸಂಶೋಧನೆಗಳು:' ಮತ್ತು "
            "IMPRESSION ಬದಲು 'ಅಭಿಪ್ರಾಯ:' ಎಂಬ ಶೀರ್ಷಿಕೆ ಬಳಸಿ."
        ),
        "disclaimer": "ಇದು AI ರಚಿಸಿದ ಎರಡನೇ ಅಭಿಪ್ರಾಯ, ಪ್ರಾಥಮಿಕ ರೋಗನಿರ್ಣಯ ಅಲ್ಲ.",
        "direction": "ltr",
    },
    "ml": {
        "name": "Malayalam",
        "script": "Malayalam",
        "findings_header": "കണ്ടെത്തലുകൾ",
        "impression_header": "അഭിപ്രായം",
        "instruction": (
            "മുഴുവൻ റിപ്പോർട്ടും മലയാളത്തിൽ എഴുതുക. "
            "വൈദ്യശാസ്ത്ര പദങ്ങൾ ലളിതമായ മലയാളത്തിൽ വിശദീകരിക്കുക. "
            "FINDINGS-ന് പകരം 'കണ്ടെത്തലുകൾ:' ഉം "
            "IMPRESSION-ന് പകരം 'അഭിപ്രായം:' ഉം ഉപയോഗിക്കുക."
        ),
        "disclaimer": "ഇത് AI സൃഷ്ടിച്ച ദ്വിതീയ അഭിപ്രായമാണ്, പ്രാഥമിക നിർണ്ണയമല്ല.",
        "direction": "ltr",
    },
    "mr": {
        "name": "Marathi",
        "script": "Devanagari",
        "findings_header": "निष्कर्ष",
        "impression_header": "निदान",
        "instruction": (
            "संपूर्ण अहवाल मराठीत लिहा. "
            "वैद्यकीय शब्दावली मराठीत स्पष्ट करा. "
            "FINDINGS ऐवजी 'निष्कर्ष:' आणि "
            "IMPRESSION ऐवजी 'निदान:' असे शीर्षक वापरा."
        ),
        "disclaimer": "हे AI-निर्मित द्वितीय मत आहे, प्राथमिक निदान नाही.",
        "direction": "ltr",
    },
    "bn": {
        "name": "Bengali",
        "script": "Bengali",
        "findings_header": "ফলাফল",
        "impression_header": "মতামত",
        "instruction": (
            "সম্পূর্ণ প্রতিবেদনটি বাংলায় লিখুন। "
            "চিকিৎসা পরিভাষা সহজ বাংলায় ব্যাখ্যা করুন। "
            "FINDINGS-এর পরিবর্তে 'ফলাফল:' এবং "
            "IMPRESSION-এর পরিবর্তে 'মতামত:' শিরোনাম ব্যবহার করুন।"
        ),
        "disclaimer": "এটি AI-উৎপন্ন দ্বিতীয় মতামত, প্রাথমিক রোগ নির্ণয় নয়।",
        "direction": "ltr",
    },
    "gu": {
        "name": "Gujarati",
        "script": "Gujarati",
        "findings_header": "તારણો",
        "impression_header": "અભિપ્રાય",
        "instruction": (
            "સંપૂર્ણ અહેવાલ ગુજરાતીમાં લખો. "
            "તબીબી શબ્દો સરળ ગુજરાતીમાં સમજાવો. "
            "FINDINGS ને બદલે 'તારણો:' અને "
            "IMPRESSION ને બદલે 'અભિપ્રાય:' શીર્ષક વાપરો."
        ),
        "disclaimer": "આ AI-જનરેટેડ બીજો અભિપ્રાય છે, પ્રાથમિક નિદાન નથી.",
        "direction": "ltr",
    },
    "pa": {
        "name": "Punjabi",
        "script": "Gurmukhi",
        "findings_header": "ਨਤੀਜੇ",
        "impression_header": "ਰਾਏ",
        "instruction": (
            "ਪੂਰੀ ਰਿਪੋਰਟ ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖੋ। "
            "ਡਾਕਟਰੀ ਸ਼ਬਦ ਸਰਲ ਪੰਜਾਬੀ ਵਿੱਚ ਸਮਝਾਓ। "
            "FINDINGS ਦੀ ਜਗ੍ਹਾ 'ਨਤੀਜੇ:' ਅਤੇ "
            "IMPRESSION ਦੀ ਜਗ੍ਹਾ 'ਰਾਏ:' ਸਿਰਲੇਖ ਵਰਤੋ।"
        ),
        "disclaimer": "ਇਹ AI-ਤਿਆਰ ਦੂਜੀ ਰਾਏ ਹੈ, ਮੁਢਲਾ ਨਿਦਾਨ ਨਹੀਂ।",
        "direction": "ltr",
    },
}

# Supported language codes (for validation)
SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIGS.keys())

def get_language_config(language: str) -> dict:
    """Return language config, falling back to English on unknown code."""
    return LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en"])


# ══════════════════════════════════════════════════════════
# Prompt Builders
# ══════════════════════════════════════════════════════════

def build_report_prompt(
    modality: str,
    findings: dict,
    structures=None,
    detected_region: str = None,
    language: str = "en",
) -> str:
    """Build a modality-specific, language-aware prompt for report generation."""
    lang = get_language_config(language)
    template = MODALITY_TEMPLATES.get(modality, DEFAULT_TEMPLATE)

    findings_text = json.dumps(findings, indent=2)
    if structures is None:
        structures_text = "None detected"
    elif isinstance(structures, dict):
        structures_text = json.dumps(structures, indent=2)
    else:
        structures_text = json.dumps(structures, indent=2)
    region_text = detected_region or "Not specified"
    findings_hdr = lang["findings_header"]
    impression_hdr = lang["impression_header"]
    disclaimer = lang["disclaimer"]
    lang_instruction = lang["instruction"]

    base_prompt = template.format(
        findings=findings_text,
        structures=structures_text,
        detected_region=region_text,
    )

    # Append language-specific output instructions
    return (
        f"{base_prompt}\n\n"
        f"--- LANGUAGE REQUIREMENT ---\n"
        f"{lang_instruction}\n"
        f"Use exactly these section headers:\n"
        f"  '{findings_hdr}:'\n"
        f"  '{impression_hdr}:'\n"
        f"End with: \"{disclaimer}\""
    )


def build_system_prompt(language: str = "en") -> str:
    """Build the LLM system prompt with language instruction."""
    lang = get_language_config(language)
    is_english = language == "en"

    base = (
        "You are a senior radiologist at a teaching hospital in India. "
        "Generate a professional radiology report in standard format: "
        f"'{lang['findings_header']}' section (detailed observations) and "
        f"'{lang['impression_header']}' section (brief clinical summary with recommendations). "
        "Use clear, precise medical language. "
        f"Always note that this is an AI-generated second opinion."
    )

    if not is_english:
        base += (
            f" CRITICAL: Write your ENTIRE response in {lang['name']} "
            f"using {lang['script']} script. "
            f"Do NOT use English except for established medical acronyms (CT, MRI, ECG, etc.)."
        )

    return base


# ══════════════════════════════════════════════════════════
# Modality Templates (language-agnostic structure,
# language instructions are appended by build_report_prompt)
# ══════════════════════════════════════════════════════════

DEFAULT_TEMPLATE = """Generate a professional radiology second-opinion report.

MODALITY: General Medical Imaging
DETECTED REGION: {detected_region}

AI FINDINGS (raw model output):
{findings}

STRUCTURES DETECTED:
{structures}

Generate the report with all key observations and a brief clinical summary."""


MODALITY_TEMPLATES = {
    "xray": """Generate a professional radiology report for this X-ray analysis.

BODY REGION (auto-detected): {detected_region}

AI FINDINGS (raw model output):
{findings}

STRUCTURES IDENTIFIED:
{structures}

Systematically describe all observations. For chest X-rays: comment on heart size, lung fields, mediastinum, bones, soft tissues. For bone X-rays: comment on fractures, alignment, joint spaces. For abdominal X-rays: comment on bowel gas, calcifications, free air. Provide differential diagnosis and recommendations.""",

    "brain_mri": """Generate a professional neuroradiology report for this brain MRI (India tertiary-care context).

AI FINDINGS:
{findings}

STRUCTURES SEGMENTED:
{structures}

INDIA / CLINICAL CONTEXT:
- Stroke and neurotuberculosis remain important differentials; correlate with clinical presentation and, when available, diffusion/ADC and post-contrast patterns.
- Neurocysticercosis: consider in seizure presentations with classic ring lesions — avoid over-calling without a compatible imaging pattern.
- Traumatic brain injury and hypertensive hemorrhage are common referrals; automated volumes and midline metrics are adjuncts only.

Systematically review: brain parenchyma, ventricles, extra-axial spaces, vascular territories, and posterior fossa. State limitations of automated volumetrics. Provide differential diagnosis and follow-up recommendations.""",

    "ecg": """Generate a professional ECG interpretation report.

AI FINDINGS:
{findings}

PARAMETERS DETECTED:
{structures}

Comment on: heart rate, rhythm, axis, intervals (PR, QRS, QT/QTc), ST-T wave abnormalities, chamber enlargement signs. State clinical significance and recommendations.""",

    "mammography": """Generate a professional mammography report.

AI FINDINGS:
{findings}

DETECTED STRUCTURES:
{structures}

Comment on: breast composition, masses, calcifications, asymmetries, lymph nodes. Include BI-RADS category with reasoning and recommendation.""",

    "pathology": """Generate a professional surgical pathology report.

AI FINDINGS:
{findings}

CLASSIFICATIONS:
{structures}

Comment on: tissue type, architecture, cellular features. State diagnosis with tumor grade/stage if applicable.""",

    "oral_cancer": """Generate an oral pathology screening report for Indian clinical context (clinical photograph screening, not histopathology).

AI FINDINGS (JSON may include items, pathology_scores, impression, clinical_notes):
{findings}

STRUCTURES / SCORES:
{structures}

INDIA CLINICAL CONTEXT:
- Primary risk factors: tobacco chewing, gutka, betel nut with tobacco (pan masala), smoking.
- Oral submucous fibrosis (OSMF) is a common OPMD pattern — note limited mouth opening if clinically relevant.
- OPMD prevalence is high in ages ~30–50 with tobacco use; OSCC-suspicious lesions need urgent referral.

REPORT STRUCTURE:
1. Lesion / mucosa description (color, borders, location if inferable from context).
2. Risk classification aligned to Normal vs OPMD vs OSCC-suspicious with probabilities.
3. If clinical_notes mention tobacco/betel, incorporate into risk stratification.
4. Next steps: routine vs short-interval follow-up vs urgent biopsy / ENT–oral surgery referral.
5. Avoid claiming histopathology diagnosis — this is a screening photograph.""",

    "abdominal_ct": """Generate a professional abdominal CT report.

AI FINDINGS:
{findings}

ORGANS SEGMENTED:
{structures}

Systematically review: liver, spleen, kidneys, pancreas, adrenals, bowel, lymph nodes, vasculature, bones. Include significant findings and recommendations.""",

    "cardiac_ct": """Generate a professional cardiac CT report.

AI FINDINGS:
{findings}

CARDIAC STRUCTURES:
{structures}

Comment on: chamber volumes, wall thickness, valvular structures, coronary arteries, pericardium, aorta. State cardiac function assessment.""",

    "ultrasound": """Generate a professional ultrasound report.

AI FINDINGS:
{findings}

STRUCTURES IDENTIFIED:
{structures}

Describe organ measurements, echogenicity, Doppler findings. Provide clinical correlation recommendations.""",

    "spine_neuro": """Generate a professional spine imaging report (cervical/thoracic/lumbar as applicable; India clinical context).

AI FINDINGS:
{findings}

VERTEBRAE/STRUCTURES:
{structures}

INDIA / CLINICAL CONTEXT:
- Degenerative disc disease and spondylosis are extremely common; separate incidental age-related change from neural compression.
- Tuberculous spondylitis (Pott disease) remains relevant — note disc involvement, endplate destruction, paraspinal/abscess pattern when inferable from findings.
- Traumatic fractures and osteoporosis-related compression fractures are frequent — align height loss and alignment comments with level-wise output when present.

Comment on: alignment, vertebral body heights, disc spaces, canal and foraminal dimensions, cord/cauda findings if stated, paraspinal soft tissues. State clinical significance and when advanced imaging or surgical referral is appropriate.""",

    "cytology": """Generate a professional cytology report.

AI FINDINGS:
{findings}

CELL CLASSIFICATIONS:
{structures}

Comment on: cellularity, cell types, nuclear features, background. State Bethesda category (Pap) or cytologic diagnosis with recommendations.""",

    "lab_report": """Generate a professional clinical pathology interpretation for an Indian tertiary-care context.

# Reference ranges below reflect Indian population studies and WHO/ICMR guidelines — verify against your laboratory's reference intervals.

AI FINDINGS (JSON):
{findings}

PARAMETERS / STRUCTURES:
{structures}

DETECTED REGION CONTEXT: {detected_region}

Cover where applicable:
- CBC: comment on anaemia pattern; **thalassaemia trait** (common in India): low MCV with low MCH and normal/high RBC — do not label as iron deficiency without ferritin; peripheral smear where relevant.
- LFT / RFT / electrolytes: renal/hepatic patterns; K+ / Na+ emergencies when values are extreme.
- Lipid panel and cardiovascular risk framing for Indian populations.
- Thyroid panel (TSH, free T4/T3) and pregnancy/biomarker context if data present.
- HbA1c: diabetes vs pre-diabetes framing; note haemoglobinopathies may skew HbA1c.
- Vitamin B12 / folate: common deficiency in vegetarian diets — interpret with diet context.
- Coagulation (INR) if present; cardiac markers (troponin, BNP) if present.

Structure the narrative as: (1) summary of tests received, (2) panel-wise interpretation, (3) integrated clinical impression, (4) urgency (routine vs urgent vs emergency), (5) suggested follow-up tests or referrals.

Comment on all abnormal values, clinical significance, and follow-up actions.""",
}


# ══════════════════════════════════════════════════════════
# Unified Report Prompt
# ══════════════════════════════════════════════════════════

def build_unified_report_prompt(
    individual_reports: list,
    language: str = "en",
    correlations_block: str = "",
) -> str:
    """Build a cross-modality unified analysis prompt."""
    lang = get_language_config(language)

    reports_text = ""
    for i, report in enumerate(individual_reports):
        modality = report.get("modality", "Unknown").upper()
        findings = report.get("findings_summary", "No findings available")
        impression = report.get("impression", "No impression available")
        reports_text += (
            f"\n--- MODALITY {i+1}: {modality} ---\n"
            f"FINDINGS: {findings}\n"
            f"IMPRESSION: {impression}\n"
        )

    corr_section = ""
    if correlations_block:
        corr_section = (
            f"\n\nPRE-IDENTIFIED CROSS-MODALITY CORRELATIONS (rules engine):\n{correlations_block}\n"
            f"Incorporate these into CROSS_MODALITY_CORRELATIONS and clinical reasoning where appropriate.\n"
        )

    lang_instruction = ""
    if language != "en":
        lang_instruction = (
            f"\n\nCRITICAL: Write your ENTIRE response in {lang['name']} "
            f"using {lang['script']} script."
        )

    return (
        f"You are a senior radiologist at a tertiary care hospital in India.\n"
        f"You have received individual reports from {len(individual_reports)} modalities for the SAME patient.\n"
        f"Perform a comprehensive cross-modality unified analysis.\n"
        f"{lang_instruction}\n\n"
        f"INDIVIDUAL MODALITY REPORTS:\n{reports_text}\n"
        f"{corr_section}\n"
        f"Output EACH section with the exact headers shown below:\n\n"
        f"UNIFIED_DIAGNOSIS:\n"
        f"Comprehensive diagnosis integrating all modalities.\n\n"
        f"UNIFIED_FINDINGS:\n"
        f"Key findings across all modalities.\n\n"
        f"CROSS_MODALITY_CORRELATIONS:\n"
        f"Specific correlations between modalities.\n\n"
        f"RISK_ASSESSMENT:\n"
        f"Comprehensive risk assessment based on combined evidence.\n\n"
        f"TREATMENT_RECOMMENDATIONS:\n"
        f"Immediate actions, management plan, follow-up, referrals.\n\n"
        f"PROGNOSIS:\n"
        f"Expected course and follow-up timeline.\n\n"
        f"CONFIDENCE:\n"
        f"Overall confidence level (low/moderate/high) with reasoning.\n\n"
        f"End with: \"{lang['disclaimer']}\""
    )
