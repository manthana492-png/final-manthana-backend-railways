"""Manthana — Cross-modality correlation rules (pattern matching before LLM)."""

from __future__ import annotations

import re
from typing import Any


def _flatten_result(modality: str, result: dict) -> dict[str, Any]:
    """Normalize modality result into comparable scalar fields."""
    out: dict[str, Any] = {"modality": modality}
    ps = result.get("pathology_scores") or {}
    for k, v in ps.items():
        key = f"{modality}.{k}"
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            out[key] = v
    # Common aliases (rules use flat keys xray.*; chest_xray / body_xray alias here)
    if modality in ("xray", "body_xray", "chest_xray", "cxr"):
        out["xray.pleural_effusion"] = float(ps.get("pleural_effusion", ps.get("Pleural_Effusion", 0)) or 0)
        out["xray.lung_lesion"] = float(ps.get("lung_lesion", ps.get("Lung_Lesion", 0)) or 0)
        out["xray.mass"] = float(ps.get("mass", ps.get("Mass", 0)) or 0)
        out["xray.fracture"] = float(ps.get("fracture", ps.get("Fracture", 0)) or 0)
        out["xray.cardiomegaly"] = float(ps.get("cardiomegaly", 0) or 0)
        out["xray.pneumonia"] = float(ps.get("pneumonia", 0) or 0)
        out["xray.consolidation"] = float(ps.get("consolidation", 0) or 0)
        out["xray.nodule_confidence"] = float(
            ps.get("nodule_confidence", ps.get("lung_nodule", ps.get("nodule", 0))) or 0
        )
        inf = float(ps.get("infiltrate_confidence", ps.get("pneumonia", ps.get("consolidation", 0))) or 0)
        out["cxr.infiltrate_confidence"] = inf
    if modality == "cxr":
        for k, v in ps.items():
            try:
                out[f"xray.{k}"] = float(v)
            except (TypeError, ValueError):
                pass
        inf_c = float(ps.get("infiltrate_confidence", ps.get("pneumonia", ps.get("consolidation", 0))) or 0)
        out["cxr.infiltrate_confidence"] = inf_c
    if modality == "lab_report":
        # optional structured labs in result — extract numeric value for rule matching
        labs = result.get("labs") or result.get("structured") or {}
        if isinstance(labs, dict):
            for lk, lv in labs.items():
                key = f"lab_report.{lk}"
                if isinstance(lv, dict) and "value" in lv:
                    try:
                        out[key] = float(lv["value"])
                    except (TypeError, ValueError):
                        out[key] = lv
                else:
                    try:
                        out[key] = float(lv)
                    except (TypeError, ValueError):
                        out[key] = lv
        st_lab = result.get("structures")
        tr_list: list[Any] = []
        if isinstance(st_lab, dict):
            tr_raw = st_lab.get("test_results")
            if isinstance(tr_raw, list):
                tr_list = tr_raw
        esr_elev = 0.0
        hbsag_r = 0.0
        for r in tr_list:
            if not isinstance(r, dict):
                continue
            tn = str(r.get("test_name") or "").upper()
            fl = str(r.get("flag") or "").upper()
            if ("ESR" in tn or "CRP" in tn or "C-REACTIVE" in tn) and fl in ("H", "HH", "CRITICAL"):
                esr_elev = 1.0
            if "HBSAG" in tn and "REACTIVE" in str(r.get("value") or "").upper():
                hbsag_r = 1.0
        out["lab.esr_elevated"] = esr_elev
        out["lab.hbsag_reactive"] = hbsag_r
    if modality == "abdominal_ct":
        out["abdominal_ct.bmd_score"] = result.get("pathology_scores", {}).get("bmd_score")
        lf = result.get("pathology_scores", {}).get("low_bmd_flag")
        if lf is not None:
            out["abdominal_ct.low_bmd_flag"] = float(lf)
        lc = ps.get("liver_cm3")
        if lc is not None:
            try:
                out["abdominal_ct.liver_cm3"] = float(lc)
            except (TypeError, ValueError):
                out["abdominal_ct.liver_cm3"] = 0.0
        af = ps.get("aaa_flag")
        if af is not None:
            try:
                out["abdominal_ct.aaa_flag"] = float(af)
            except (TypeError, ValueError):
                pass
        llc = ps.get("liver_lesion_confidence")
        if llc is not None:
            try:
                out["abdominal_ct.liver_lesion_confidence"] = float(llc)
            except (TypeError, ValueError):
                pass
    if modality == "cardiac_ct":
        lv = ps.get("lv_cm3")
        if lv is not None:
            try:
                out["cardiac_ct.lv_cm3"] = float(lv)
            except (TypeError, ValueError):
                out["cardiac_ct.lv_cm3"] = 0.0
    if modality == "pathology":
        ms = float(ps.get("malignancy_score", 0) or 0)
        out["pathology.malignancy_score"] = ms
        out["pathology.malignancy_confidence"] = float(
            ps.get("malignancy_confidence", ms) or 0
        )
    if modality == "cytology":
        st = result.get("structures")
        if isinstance(st, dict) and st.get("is_critical") is not None:
            out["cytology.is_critical_struct"] = 1.0 if st.get("is_critical") else 0.0
        hci = ps.get("hsil_confidence")
        if hci is None and isinstance(ps.get("HSIL"), (int, float)):
            hci = ps.get("HSIL")
        if hci is not None:
            try:
                out["cytology.hsil_confidence"] = float(hci)
            except (TypeError, ValueError):
                pass
    if modality == "dermatology":
        st = result.get("structures")
        if isinstance(st, dict):
            tc = st.get("top_condition")
            if tc is not None and str(tc).strip():
                out["dermatology.top_condition"] = str(tc).strip().lower()
    if modality == "mammography":
        st = result.get("structures")
        if isinstance(st, dict):
            out["mammography.has_four_views"] = 1.0 if st.get("has_four_views") else 0.0
            rc = st.get("risk_category")
            if rc is not None and str(rc).strip():
                out["mammography.risk_category"] = str(rc).strip().lower()
        out["mammography.birads_4_or_above"] = float(ps.get("birads_4_or_above", 0) or 0)
        out["mammography.malignancy_confidence"] = float(ps.get("malignancy_confidence", 0) or 0)
        out["mammography.is_high_risk"] = float(ps.get("is_high_risk", 0) or 0)
    if modality in ("spine_neuro", "spine_mri"):
        alt = "spine_mri" if modality == "spine_neuro" else "spine_neuro"
        for k, v in ps.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = v
            out[f"{modality}.{k}"] = fv
            out[f"{alt}.{k}"] = fv
    if modality == "ecg":
        # Ensure QTc / age available for rules even if only in structures
        st = result.get("structures")
        if isinstance(st, dict):
            iv = st.get("intervals") or {}
            if iv.get("qtc_ms") is not None and "ecg.qtc_ms" not in out:
                try:
                    out["ecg.qtc_ms"] = float(iv["qtc_ms"])
                except (TypeError, ValueError):
                    pass
    if modality in ("ultrasound", "usg", "us"):
        ps = result.get("pathology_scores") or {}
        out["usg.free_fluid_present"] = float(ps.get("free_fluid_present", False) or 0.0)
        out["usg.ascites_indicator"] = float(ps.get("ascites_indicator", 0.0) or 0.0)
        out["usg.liver_echogenicity_high"] = float(
            ps.get("liver_echogenicity_high", False) or 0.0
        )
        out["usg.renal_echogenicity_high"] = float(
            ps.get("renal_echogenicity_high", False) or 0.0
        )
        out["usg.parenchymal_heterogeneity"] = float(
            ps.get("parenchymal_heterogeneity", 0.0) or 0.0
        )
        out["usg.anomaly_proxy_score"] = float(
            ps.get("anomaly_proxy_score", 0.0) or 0.0
        )
        out["usg.image_quality_adequate"] = float(
            ps.get("image_quality_adequate", True) or 0.0
        )
        findings_text = str(result.get("findings", "")).lower()
        out["usg.gallstones_mentioned"] = 1.0 if any(
            w in findings_text
            for w in (
                "gallstone",
                "cholelithiasis",
                "choledocholithiasis",
                "biliary calculi",
                "gall bladder stone",
            )
        ) else 0.0
        out["usg.hepatomegaly_mentioned"] = 1.0 if any(
            w in findings_text
            for w in ("hepatomegaly", "enlarged liver", "liver span")
        ) else 0.0
        out["usg.hydronephrosis_mentioned"] = 1.0 if any(
            w in findings_text
            for w in (
                "hydronephrosis",
                "pelvicalyceal",
                "ureteric calculi",
                "renal calculi",
            )
        ) else 0.0
        out["usg.lymphadenopathy_mentioned"] = 1.0 if any(
            w in findings_text
            for w in (
                "lymph node",
                "lymphadenopathy",
                "mesenteric node",
                "retroperitoneal",
            )
        ) else 0.0
    return out


def _eval_condition(value: Any, cond: str) -> bool:
    cond = cond.strip()
    if cond.lower() in ("true", "1", "yes"):
        return bool(value)
    if cond.lower() in ("false", "0", "no"):
        return not bool(value)
    meq = re.match(r"^==\s*([0-9.+-]+)$", cond)
    if meq and isinstance(value, (int, float)):
        return float(value) == float(meq.group(1))
    m = re.match(r"^([><=]+)\s*([0-9.+-]+)$", cond)
    if m and isinstance(value, (int, float)):
        op, rhs = m.group(1), float(m.group(2))
        if op == ">":
            return float(value) > rhs
        if op == "<":
            return float(value) < rhs
        if op == ">=":
            return float(value) >= rhs
        if op == "<=":
            return float(value) <= rhs
        if op == "==":
            return float(value) == rhs
    if " OR " in cond.upper():
        parts = re.split(r"\s+OR\s+", cond, flags=re.IGNORECASE)
        return any(_eval_condition(value, p.strip()) for p in parts)
    return False


def _match_requires(flat: dict[str, Any], requires: dict[str, dict[str, str]]) -> tuple[int, int]:
    matched = 0
    total = 0
    for mod, fields in requires.items():
        for field, cond in fields.items():
            key = f"{mod}.{field}" if "." not in field else field
            total += 1
            val = flat.get(key)
            if val is None:
                continue
            try:
                if _eval_condition(val, cond):
                    matched += 1
            except Exception:
                continue
    return matched, total


CORRELATION_RULES: list[dict[str, Any]] = [
    {
        "name": "Heart Failure Indicators",
        "requires": {
            "xray": {"pleural_effusion": ">0.5"},
            "lab_report": {"BNP": ">400"},
            "cardiac_ct": {"lv_cm3": ">200"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Cardiology referral recommended",
    },
    {
        "name": "Metastatic Disease Pattern",
        "requires": {
            "xray": {"lung_lesion": ">0.5"},
            "abdominal_ct": {"liver_cm3": ">1800"},
            "pathology": {"malignancy_score": ">0.6"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": "Oncology tumor board review recommended",
    },
    {
        "name": "CT Low BMD — Osteoporosis Risk (India context)",
        "requires": {
            "abdominal_ct": {"low_bmd_flag": "==1"},
        },
        "min_match": 1,
        "clinical_significance": "high",
        "action": "DEXA scan recommended. Consider vitamin D and dietary calcium assessment where appropriate.",
    },
    {
        "name": "CT Aortic Aneurysm — Elevated AAA Flag",
        "requires": {
            "abdominal_ct": {"aaa_flag": ">0"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": "Vascular surgery referral if clinically indicated; correlate with diameter and symptoms.",
    },
    {
        "name": "CT Hepatomegaly Pattern with CXR Effusion",
        "requires": {
            "abdominal_ct": {"liver_cm3": ">1800"},
            "xray": {"pleural_effusion": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Correlate clinically; consider infectious/hepatic workup as indicated.",
    },
    {
        "name": "Osteoporotic Fracture Risk",
        "requires": {
            "abdominal_ct": {"low_bmd_flag": "==1"},
            "xray": {"fracture": ">0.3"},
        },
        "min_match": 2,
        "clinical_significance": "warning",
        "action": "Endocrinology referral. DXA scan recommended.",
    },
    {
        "name": "ECG LVH with CXR Cardiomegaly",
        "requires": {
            "ecg": {"lvh": ">0.5"},
            "xray": {"cardiomegaly": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Cardiology referral. Echocardiography recommended.",
    },
    {
        "name": "ECG Atrial Fibrillation with CXR Cardiomegaly",
        "requires": {
            "ecg": {"atrial_fibrillation": ">0.5"},
            "xray": {"cardiomegaly": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Urgent cardiology review. Anticoagulation assessment.",
    },
    {
        "name": "ECG LBBB with CXR Cardiomegaly",
        "requires": {
            "ecg": {"lbbb": ">0.5"},
            "xray": {"cardiomegaly": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Urgent evaluation. Treat as STEMI equivalent if new onset.",
    },
    {
        "name": "Oral OSCC High Risk (screening scores)",
        "requires": {
            "oral_cancer": {"oscc_suspicious": ">0.5"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": "Urgent biopsy referral. India: document tobacco/betel exposure when known; correlate clinically.",
    },
    {
        "name": "Oral OPMD — Premalignant Monitoring",
        "requires": {
            "oral_cancer": {"opmd": ">0.5"},
        },
        "min_match": 1,
        "clinical_significance": "high",
        "action": "Short-interval follow-up and clinical oral exam; OSMF/cessation counselling where tobacco use is reported.",
    },
    {
        "name": "STEMI Pattern (ECG + troponin)",
        "requires": {
            "ecg": {"st_elevation": ">0.5"},
            "lab_report": {"troponin_i": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": "ST elevation pattern with elevated troponin — emergency cardiology; correlate clinically.",
    },
    {
        "name": "Long QT — Drug / Electrolyte Check",
        "requires": {
            "ecg": {"qtc_ms": ">500"},
        },
        "min_match": 1,
        "clinical_significance": "warning",
        "action": (
            "QTc >500ms — check all QT-prolonging medications "
            "(azithromycin, haloperidol, ondansetron, chloroquine). "
            "Urgent electrolyte panel: K+, Mg2+, Ca2+. "
            "Hypokalemia is leading cause in India. "
            "Cardiology review same day."
        ),
        "modalities_involved": ["ecg", "lab_report"],
    },
    {
        "name": "AFib Young Patient — RHD Screening",
        "requires": {
            "ecg": {"afib_confidence": ">0.7", "patient_age": "<45"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "AF in patient under 45 — Rheumatic Heart Disease "
            "must be excluded. Echocardiography mandatory. "
            "Mitral stenosis is leading cause of AF in young "
            "Indians. RHD prevalence 1.5-2% in endemic areas."
        ),
        "modalities_involved": ["ecg"],
    },
    {
        "name": "Significant Anaemia (lab)",
        "requires": {
            "lab_report": {"hemoglobin": "<8"},
        },
        "min_match": 1,
        "clinical_significance": "warning",
        "action": "Significant anaemia — iron studies, B12, folate, peripheral smear as indicated.",
    },
    {
        "name": "CKD Pattern (creatinine + urea)",
        "requires": {
            "lab_report": {"creatinine": ">10", "urea": ">15"},
        },
        "min_match": 2,
        "clinical_significance": "warning",
        "action": "Renal impairment pattern — eGFR, nephrology referral if progressive; correlate volume status and medications.",
    },
    {
        "name": "Derm Tinea with Elevated Glucose",
        "requires": {
            "dermatology": {"tinea": ">0.5"},
            "lab_report": {"glucose": ">7"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": (
            "Tinea pattern with elevated glucose — optimise glycaemic control; treat fungal infection; "
            "rule out steroid-modified tinea if topical steroids were used."
        ),
    },
    {
        "name": "Derm Vitiligo with High TSH",
        "requires": {
            "dermatology": {"vitiligo": ">0.5"},
            "lab_report": {"tsh": ">4.5"},
        },
        "min_match": 2,
        "clinical_significance": "warning",
        "action": (
            "Vitiligo with elevated TSH — consider autoimmune thyroid disease; anti-TPO/anti-TG if clinically indicated."
        ),
    },
    {
        "name": "Derm Vitiligo with Low TSH",
        "requires": {
            "dermatology": {"vitiligo": ">0.5"},
            "lab_report": {"tsh": "<0.4"},
        },
        "min_match": 2,
        "clinical_significance": "warning",
        "action": (
            "Vitiligo with suppressed TSH — correlate for hyperthyroidism; endocrine follow-up if symptomatic."
        ),
    },
    {
        "name": "Derm Psoriasis with Elevated ESR",
        "requires": {
            "dermatology": {"psoriasis": ">0.5"},
            "lab_report": {"esr": ">40"},
        },
        "min_match": 2,
        "clinical_significance": "warning",
        "action": (
            "Psoriasis with raised ESR — screen for psoriatic arthritis and systemic inflammation as clinically appropriate."
        ),
    },

    {
        "name": "Cytology Critical — Urgent Follow-up",
        "requires": {
            "cytology": {"is_critical": ">0.5"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "Critical cytology pattern (e.g. HSIL/AIS/malignant-equivalent on AI screening). "
            "URGENT gynaecology/colposcopy referral for cervical cytology as clinically appropriate; "
            "correlate with HPV testing where indicated. India: cervical cancer burden is high — do not delay."
        ),
    },
    {
        "name": "Cytology and Histology Concordant Malignancy Signal",
        "requires": {
            "cytology": {"malignant": ">0.5"},
            "pathology": {"malignancy_score": ">0.6"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "Cytology and histology both suggest elevated malignancy risk on AI screening — "
            "oncology/tumour board referral and staging workup as clinically indicated."
        ),
    },
    {
        "name": "Sputum Cytology TB Pattern (Screening)",
        "requires": {
            "cytology": {"tb_suggestive": ">0.4"},
        },
        "min_match": 1,
        "clinical_significance": "high",
        "action": (
            "Sputum cytology suggests TB-associated morphology on AI screening — "
            "send sputum for AFB/GeneXpert as per local protocol; infection control and clinical correlation."
        ),
    },
    {
        "name": "Derm Psoriasis with Elevated CRP",
        "requires": {
            "dermatology": {"psoriasis": ">0.5"},
            "lab_report": {"crp": ">10"},
        },
        "min_match": 2,
        "clinical_significance": "warning",
        "action": (
            "Psoriasis with raised CRP — consider inflammatory comorbidity; musculoskeletal review if joint symptoms."
        ),
    },
    {
        "name": "Mammography High Risk — Specialist Review",
        "requires": {
            "mammography": {"is_high_risk": ">0.5"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "Mirai 5-year risk above 5% threshold — refer to breast surgery/oncology per protocol. "
            "Discuss risk reduction; consider supplemental MRI for dense breasts. "
            "India: follow local high-risk surveillance guidelines (e.g. IAC/FOGSI where applicable)."
        ),
    },
    {
        "name": "Mammography + Pathology Concordant Malignancy",
        "requires": {
            "mammography": {"is_high_risk": ">0.5"},
            "pathology": {"malignancy_score": ">0.6"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "Elevated mammography risk flag and histology malignancy score on AI screening — "
            "concordant multi-modal signal. Urgent oncology referral for staging workup as clinically indicated."
        ),
    },
    {
        "name": "Pott's Disease + Pulmonary TB — RNTCP Protocol",
        "requires": {
            "spine_neuro": {"pott_disease_confidence": ">0.6"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "Vertebral TB (Pott's disease) suspected on MRI. "
            "CT chest mandatory to exclude pulmonary TB co-infection "
            "(concurrent in ~50% of Pott's cases in India). "
            "CT guided biopsy of paravertebral lesion for AFB culture "
            "and histopathology (confirm Mycobacterium tuberculosis). "
            "RNTCP/NTEP regimen: 2HRZE + 10HR for spinal TB. "
            "Neurosurgical referral if cord compression present. "
            "India: Bihar, UP, Maharashtra — highest Pott's prevalence. "
            "Notify district TB officer (NIKSHAY registration mandatory)."
        ),
        "modalities_involved": ["spine_neuro", "cxr"],
    },
    {
        "name": "Cord Compression — Neurosurgical Emergency",
        "requires": {
            "spine_neuro": {"cord_compression_confidence": ">0.65"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "Spinal cord compression on MRI — neurosurgical emergency. "
            "Immediate neurosurgery or orthopedic spine referral. "
            "Do not delay for further workup if myelopathy present. "
            "IV methylprednisolone protocol if acute traumatic cause. "
            "India: AIIMS, Nimhans, Sree Chitra — national spinal "
            "cord injury centres. Same-day transfer if rural setting."
        ),
        "modalities_involved": ["spine_neuro"],
    },
    {
        "name": "BI-RADS 4C/5 — Core Needle Biopsy Mandatory",
        "requires": {
            "mammography": {"birads_4_or_above": ">0.8"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "BI-RADS 4C or 5 mammographic finding — biopsy mandatory. "
            "Do not repeat mammogram or wait. "
            "Ultrasound-guided 14G core needle biopsy preferred. "
            "If mass non-visible on US: stereotactic biopsy. "
            "India: Kidwai Memorial, Tata Memorial, AIIMS oncology MDT. "
            "Dense breast (ACR C/D common in Indian women age 40-55) "
            "may mask additional lesions — MRI breast recommended "
            "if core biopsy positive for malignancy. "
            "NRGCP protocol for tissue banking if available."
        ),
        "modalities_involved": ["mammography"],
    },
    {
        "name": "Breast Mass + CXR Nodule — Metastasis Staging",
        "requires": {
            "mammography": {"malignancy_confidence": ">0.65"},
            "xray": {"nodule_confidence": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "Suspicious breast mass with pulmonary nodule on CXR — "
            "pulmonary metastasis must be excluded before local treatment. "
            "CECT chest + abdomen + pelvis for full staging. "
            "PET-CT if stage 3 or 4 suspected. "
            "Bone scan if bone pain or elevated ALP. "
            "India: breast cancer is most common female malignancy; "
            "median age of presentation is 47 years (vs 62 in West) "
            "— aggressive biology common. "
            "Refer oncology MDT before any surgical decision."
        ),
        "modalities_involved": ["mammography", "cxr"],
    },
    {
        "name": "CXR Pneumonia or consolidation — infection workup",
        "requires": {
            "xray": {"pneumonia": ">0.5"},
        },
        "min_match": 1,
        "clinical_significance": "high",
        "action": (
            "Elevated pneumonia probability on chest imaging — correlate clinically; consider infection workup "
            "and local TB endemicity where relevant (India context)."
        ),
    },
    {
        "name": "Brain MRI — ring-enhancing lesion signal (NCC/TB/tumour differential)",
        "requires": {
            "brain_mri": {"ncc_confidence": ">0.5"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "Elevated NCC probability on AI screening — correlate with endemic exposure, cysticercosis serology, "
            "and contrast MRI; TB tuberculoma and pyogenic abscess remain key differentials in India."
        ),
    },
    {
        "name": "Brain MRI — midline shift / mass effect concern",
        "requires": {
            "brain_mri": {"midline_shift_mm": ">5"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "Midline shift or mass effect pattern on AI metrics — urgent neurosurgical / emergency review; "
            "do not delay for definitive imaging."
        ),
    },
    {
        "name": "Cervical HSIL — Colposcopy Protocol",
        "requires": {
            "cytology": {"hsil_confidence": ">0.6"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "HSIL on cervical cytology — colposcopy mandatory. "
            "Do not repeat Pap smear; refer directly. "
            "India HPV prevalence: HPV 16/18 in ~85% of cervical cancers. "
            "LEEP/CKC if confirmed CIN2+. "
            "Refer gynecological oncology same week."
        ),
        "modalities_involved": ["cytology"],
    },
    {
        "name": "HSIL Cytology + HPV Positive — Immediate Colposcopy",
        "requires": {
            "cytology": {"hsil_confidence": ">0.5"},
            "lab_report": {"hpv_positive": ">0.8"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "HSIL cytology with HPV positivity — highest risk combination. "
            "Immediate colposcopy. Do not observe. "
            "India: co-testing not routine everywhere — "
            "VIA/VILI may be more accessible in primary care settings."
        ),
        "modalities_involved": ["cytology", "lab_report"],
    },
    {
        "name": "Adenocarcinoma + Smoking — Lung Primary Workup",
        "requires": {
            "pathology": {"malignancy_confidence": ">0.7"},
        },
        "min_match": 1,
        "clinical_significance": "critical",
        "action": (
            "Malignant pathology with tobacco exposure — "
            "lung primary must be excluded. "
            "CT chest/abdomen/pelvis for staging. "
            "India: lung cancer rising in non-smokers due to air pollution. "
            "Bidi and cigarette both high risk. "
            "Refer oncology for TNM staging and MDT discussion."
        ),
        "modalities_involved": ["pathology", "lab_report"],
    },
    {
        "name": "High Mitotic Rate + CT Liver Lesion — Metastasis Workup",
        "requires": {
            "pathology": {"malignancy_confidence": ">0.6"},
            "abdominal_ct": {"liver_lesion_confidence": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "Malignant pathology with hepatic lesion on CT — "
            "liver metastasis must be excluded. "
            "MRI liver with hepatobiliary contrast. "
            "PET-CT for full staging if MRI confirms metastasis. "
            "Refer oncology for multi-disciplinary team review."
        ),
        "modalities_involved": ["pathology", "abdominal_ct"],
    },
    {
        "name": "Lab TB Pattern + Spine Pott's — Definitive TB",
        "requires": {
            "lab_report": {"tb_pattern_confidence": ">0.6"},
            "spine_neuro": {"pott_disease_confidence": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "Lab TB pattern (pancytopaenia, lymphocytosis, high ESR/ADA) "
            "combined with vertebral TB (Pott's disease) on MRI. "
            "High probability disseminated tuberculosis. "
            "Mandatory: CT guided biopsy paravertebral tissue for "
            "AFB culture + GeneXpert MTB/RIF, histopathology. "
            "RNTCP Category I regimen: 2HRZE + 10HR. "
            "NIKSHAY registration (legally required in India). "
            "DOTS centre referral via district TB officer. "
            "Nutrition support — TB-malnutrition bidirectional. "
            "Screen household contacts within 2 weeks."
        ),
        "modalities_involved": ["lab_report", "spine_neuro"],
    },
    {
        "name": "Lab TB Pattern + CXR — Pulmonary TB Workup",
        "requires": {
            "lab_report": {"tb_pattern_confidence": ">0.6"},
            "cxr": {"infiltrate_confidence": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "Elevated ESR/CRP/ADA on labs combined with CXR infiltrate. "
            "Active pulmonary tuberculosis cannot be excluded. "
            "Mandatory: sputum AFB × 3 on consecutive days, "
            "GeneXpert MTB/RIF (preferred first-line in India), "
            "IGRA/QuantiFERON if immunocompromised. "
            "RNTCP/NTEP protocol: presume TB if GeneXpert positive — "
            "start treatment without culture confirmation if clinical "
            "score high. India-specific: Bihar, UP, Maharashtra, "
            "Jharkhand have highest TB burden. "
            "NIKSHAY registration mandatory before treatment start."
        ),
        "modalities_involved": ["lab_report", "cxr"],
    },
    {
        "name": "Diabetic Nephropathy — Urgent Nephrology Referral",
        "requires": {
            "lab_report": {
                "renal_impairment_score": ">0.6",
                "diabetes_control_score": ">0.5",
            },
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "CKD with poorly controlled diabetes confirmed on labs "
            "(eGFR < 30, HbA1c > 8%). "
            "Urgent nephrology referral for CKD staging and "
            "renoprotective therapy optimisation. "
            "Key actions: ACEi/ARB (if K+ safe), SGLT2-inhibitor "
            "(empagliflozin/dapagliflozin — now covered in CGHS), "
            "dietary protein restriction 0.6–0.8g/kg/day, "
            "dialysis access planning if eGFR < 20. "
            "India: DN is leading cause of CKD — PMJAY-covered dialysis "
            "at empanelled centres. Refer district nephrology unit."
        ),
        "modalities_involved": ["lab_report"],
    },
    {
        "name": "USG_ASCITES_LAB_ALBUMIN",
        "requires": {
            "usg": {"free_fluid_present": ">0.3"},
            "lab_report": {"albumin_low": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Free fluid on ultrasound with low serum albumin — decompensated liver disease or hypoproteinaemia likely; urgent hepatology referral recommended.",
    },
    {
        "name": "USG_LIVER_HIGH_ECHO_LAB_ALT",
        "requires": {
            "usg": {"liver_echogenicity_high": ">0.5"},
            "lab_report": {"alt_elevated": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Bright liver echotexture with elevated ALT suggests hepatic parenchymal disease (e.g., NAFLD, hepatitis); hepatology or internal-medicine follow-up recommended.",
    },
    {
        "name": "USG_FREE_FLUID_LYMPH_TB",
        "requires": {
            "usg": {"free_fluid_present": ">0.3", "lymphadenopathy_mentioned": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Ascites with mesenteric lymphadenopathy on ultrasound strongly suggests tuberculous peritonitis in endemic regions; urgent TB workup advised.",
    },
    {
        "name": "USG_HYDRONEPHROSIS_LAB_CREATININE",
        "requires": {
            "usg": {"hydronephrosis_mentioned": ">0.5"},
            "lab_report": {"creatinine_elevated": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": "Hydronephrosis on ultrasound with raised creatinine indicates obstructive nephropathy; emergency urology consult and imaging are recommended.",
    },
    {
        "name": "USG_GALLSTONES_RUQ_PAIN",
        "requires": {
            "usg": {"gallstones_mentioned": ">0.5"},
        },
        "min_match": 1,
        "clinical_significance": "high",
        "action": "Gallstones identified on ultrasound; evaluate for cholecystitis and biliary obstruction; surgical referral for cholelithiasis workup is recommended.",
    },
    {
        "name": "USG_ASCITES_CXR_PLEURAL",
        "requires": {
            "usg": {"free_fluid_present": ">0.3"},
            "cxr": {"infiltrate_confidence": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Ascites on ultrasound with pleural effusion on chest imaging suggests hepatic hydrothorax or systemic disease; urgent combined hepatology and respiratory review advised.",
    },
    {
        "name": "USG_HEPATOMEGALY_CXR_INFILTRATE",
        "requires": {
            "usg": {"hepatomegaly_mentioned": ">0.5"},
            "cxr": {"infiltrate_confidence": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "high",
        "action": "Hepatomegaly on ultrasound with pulmonary infiltrates suggests systemic infection (e.g., malaria, typhoid, miliary TB); urgent internal-medicine workup recommended.",
    },
    {
        "name": "USG_FREE_FLUID_HIGH_ANOMALY_EMERGENCY",
        "requires": {
            "usg": {
                "free_fluid_present": ">0.5",
                "parenchymal_heterogeneity": ">0.55",
            },
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": "Significant free fluid with high anomaly proxy score on ultrasound indicates acute abdominal emergency (e.g., haemoperitoneum, ruptured viscus, ectopic pregnancy); immediate surgical evaluation required.",
    },
    {
        "name": "HBsAg Reactive + CT Liver Lesion — HCC Screening",
        "requires": {
            "lab_report": {"hepatic_injury_score": ">0.7"},
            "abdominal_ct": {"liver_lesion_confidence": ">0.5"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "HBsAg reactive (confirmed Hepatitis B) with liver lesion "
            "on CT — Hepatocellular Carcinoma (HCC) must be excluded. "
            "CECT liver with arterial + portal + delayed phases. "
            "AFP tumour marker (elevated in 70% HCC). "
            "If LI-RADS 4–5: multidisciplinary HCC team (Tata Memorial, "
            "Kidwai, AIIMS liver units). "
            "India: HBV-related HCC occurs a decade earlier than West "
            "(median age 45). Treat underlying HBV first: "
            "tenofovir disoproxil or entecavir (NACO-approved regimens)."
        ),
        "modalities_involved": ["lab_report", "abdominal_ct"],
    },
    {
        "name": "Severe Anaemia + CXR Cardiomegaly — Cardiac Failure",
        "requires": {
            "lab_report": {"anaemia_severity_score": ">0.7"},
            "xray": {"cardiomegaly": ">0.4"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": (
            "Severe anaemia (Hb < 7 g/dL) with cardiomegaly on CXR — "
            "anaemia-induced high-output cardiac failure. "
            "Urgent action: Hb < 7 = blood transfusion threshold "
            "(Indian Society of Haematology guideline). "
            "ECG + echo to assess cardiac function. "
            "India: iron deficiency anaemia extremely prevalent in "
            "women of reproductive age (prevalence > 50% in rural India). "
            "IV iron (Ferric carboxymaltose) preferred over oral in "
            "severe/symptomatic cases. "
            "Rule out haemoglobinopathy (thalassaemia) in relevant "
            "ethnic groups (Gujarat, Punjab, Tamil Nadu coastal)."
        ),
        "modalities_involved": ["lab_report", "cxr"],
    },
]


def find_correlations(individual_results: list[dict]) -> list[dict]:
    """Match multi-modality results against clinical patterns."""
    flat: dict[str, Any] = {}
    modalities_present: set[str] = set()
    for item in individual_results:
        modality = item.get("modality", "")
        result = item.get("result") or {}
        modalities_present.add(modality)
        flat.update(_flatten_result(modality, result))

    correlations: list[dict] = []
    for rule in CORRELATION_RULES:
        requires = rule["requires"]
        min_match = int(rule.get("min_match", 2))
        matched, _ = _match_requires(flat, requires)
        if matched < min_match:
            continue
        mods = list(requires.keys())
        confidence = min(1.0, matched / max(min_match, 1))
        correlations.append(
            {
                "pattern": rule["name"],
                "name": rule["name"],
                "confidence": round(confidence, 2),
                "clinical_significance": rule.get("clinical_significance", "info"),
                "matching_modalities": [m for m in mods if m in modalities_present],
                "action": rule.get("action", ""),
            }
        )
    return correlations
