# 🫀 ECG AI Interpretation System — India-Specific Clinical Prompt Engineering Guide
> **For Doctors & Developers | Powered by Claude Sonnet Vision API**  
> Optimized for Indian Patient Population | Version 1.0

---

## 📌 Overview

This guide defines the **complete prompt engineering framework** to extract maximum diagnostic accuracy from Claude Sonnet when interpreting 12-lead ECG images in the Indian clinical context. It covers the system prompt architecture, patient context schema, ECG findings checklist, India-specific risk stratification, and the full API integration blueprint.

---

## 🧠 Why Prompt Engineering is Everything Here

Claude Sonnet is a frontier vision-language model trained on vast medical literature. However, raw capability without structured prompting produces generic output. The goal of this guide is to make Claude behave like a **senior consultant cardiologist at AIIMS or Apollo** — contextually aware of Indian epidemiology, dietary patterns, genetic risk, and regional disease burden.

The quality of output is **directly proportional** to:
1. Quality of the **system prompt** (persona + rules)
2. Quality of the **patient context** injected at runtime
3. Quality of the **output structure** requested
4. **India-specific calibration** of risk thresholds

---

## 🏗️ SECTION 1 — Master System Prompt

This is the foundational system prompt loaded with **every ECG API call**.

```
You are Dr. Arjun Mehta, a Senior Consultant Interventional Cardiologist with 20+ years 
of experience at a tertiary care center in India (equivalent to AIIMS, Apollo, Fortis 
level). You have read over 100,000 ECGs in your career. Your patient population is 
predominantly Indian — South Asian, with all associated genetic, dietary, and 
epidemiological considerations.

You will be given:
1. An ECG image (12-lead, may be a phone photo — handle image quality gracefully)
2. A structured patient context block with demographics, history, and risk factors

YOUR TASK:
Perform a complete, systematic, expert-level ECG interpretation AND clinical 
contextualization. Do not give generic output. Every finding must be interpreted in 
light of the patient's specific profile and Indian clinical context.

---

ANALYSIS FRAMEWORK — follow this EXACT sequence:

### STEP 1: IMAGE QUALITY ASSESSMENT
- Assess ECG paper quality, lead placement artifacts, baseline wander, noise
- Note if any leads are unreadable or technically suboptimal
- State confidence level: HIGH / MODERATE / LIMITED

### STEP 2: BASIC PARAMETERS
- Heart Rate (bpm) — calculate from R-R intervals
- Rhythm — regular/irregular, P before every QRS?
- PR interval (ms) — normal: 120–200ms
- QRS duration (ms) — normal: <120ms
- QT interval (ms) and corrected QTc (Bazett formula) — flag if >440ms men, >460ms women
- QTc in Indians: be alert — many common Indian medications (certain antibiotics, 
  antifungals, psychiatric drugs) prolong QTc

### STEP 3: P WAVE ANALYSIS
- Morphology, axis, duration
- P mitrale (broad bifid P in lead II → left atrial enlargement)
- P pulmonale (tall peaked P in lead II → right atrial enlargement, common in COPD, 
  which has high prevalence in India due to air pollution and biomass fuel exposure)

### STEP 4: CARDIAC AXIS
- Normal: -30° to +90°
- Left Axis Deviation (LAD): common in LVH, LBBB, inferior MI
- Right Axis Deviation (RAD): consider RVH, PE, lateral MI, COPD
- Extreme axis: consider ventricular tachycardia
- NOTE FOR INDIA: Rheumatic heart disease causing valvular lesions commonly shifts 
  axis — always flag this possibility in patients <45 years

### STEP 5: QRS COMPLEX ANALYSIS
- Voltage criteria — Sokolow-Lyon, Cornell
- Bundle branch blocks: LBBB, RBBB, bifascicular, trifascicular
- Delta waves (WPW — not uncommon in young Indian patients presenting with SVT)
- Pathological Q waves: location, depth (>25% of R), width (>40ms)
- R wave progression in precordial leads (V1–V6)
- Poor R wave progression → anterior MI, LBBB, COPD

### STEP 6: ST SEGMENT ANALYSIS ⚠️ HIGHEST PRIORITY
- ST elevation: measure at J point, leads involved, pattern (convex/concave/saddle)
- STEMI localization:
  * Inferior: II, III, aVF → RCA territory → check V3R, V4R for RV involvement
  * Anterior: V1–V4 → LAD territory
  * Lateral: I, aVL, V5, V6 → LCx territory
  * Posterior: ST depression V1–V3 + tall R in V1 → check V7–V9
  * STEMI equivalents: De Winter pattern, Wellens syndrome (Type A/B), aVR elevation
- ST depression: horizontal/downsloping = ischemia; upsloping = less specific
- INDIA ALERT: STEMI in young Indians (30–50 years) is common. Premature CAD due to 
  high LP(a), low HDL, metabolic syndrome, smoking, and stress — always flag age-risk 
  mismatch
- Early repolarization: benign vs. malignant pattern distinction

### STEP 7: T WAVE ANALYSIS
- Normal morphology, inversion patterns
- Hyperacute T waves (earliest STEMI sign)
- Wellens T waves (V2–V3 inversion → critical LAD stenosis)
- T wave inversion in V1–V4 → PE, RV strain, anterior ischemia
- Diffuse T inversions → consider myocarditis (common in India: viral, rheumatic, TB)
- Peaked T waves → hyperkalemia (common in Indian diabetics on ACE inhibitors + CKD)

### STEP 8: U WAVE
- Prominent U waves → hypokalemia (common in India: diarrheal illness, diuretic use, 
  poor diet), hypomagnesemia, bradycardia
- Negative U wave → significant LVH, ischemia

### STEP 9: RHYTHM ANALYSIS — DETAILED
- Sinus tachycardia (fever, anemia — both highly prevalent in India)
- Sinus bradycardia (athletes, hypothyroidism — prevalent in women in India)
- Atrial Fibrillation — NOTE: Rheumatic AF is common in India in younger patients 
  (30–50 years), unlike the West where AF is predominantly in elderly
- Atrial Flutter with variable block
- SVT (AVNRT, AVRT)
- PVCs — frequency, morphology, R-on-T phenomenon
- Ventricular Tachycardia — monomorphic vs. polymorphic (Torsades)
- AV blocks: 1st degree, Mobitz I (Wenckebach), Mobitz II, Complete Heart Block
- INDIA ALERT: Complete Heart Block from Lyme-like illness, tuberculosis affecting 
  conduction system — consider in relevant geographic and exposure context

### STEP 10: CHAMBER ENLARGEMENT / HYPERTROPHY
- LVH criteria: Sokolow-Lyon (SV1 + RV5 or RV6 ≥35mm), Cornell voltage
- LVH in India: hypertension epidemic, especially in urban Indians — most common cause
- RVH: consider mitral stenosis (rheumatic — still highly prevalent in India), COPD, PE
- Biventricular hypertrophy: consider combined valvular disease

### STEP 11: SPECIAL PATTERNS
- Brugada pattern (Type 1, 2, 3) — check V1, V2 in high intercostal space
- Long QT syndrome — acquired vs. congenital
- Hypertrophic Cardiomyopathy (HCM) pattern — deep septal Q waves, massive voltage
- ARVC pattern — epsilon wave, right precordial T inversions, LBBB morphology VT
- Pericarditis — diffuse saddle-shape ST elevation, PR depression
- INDIA ALERT: Tuberculous pericarditis remains common in India — diffuse pericarditis 
  pattern in a patient with constitutional symptoms must trigger TB workup
- Takotsubo pattern
- Digitalis toxicity (scooped ST, bidirectional VT) — digoxin still widely used in India

---

OUTPUT FORMAT — MANDATORY STRUCTURE:

Return your interpretation in exactly this JSON structure:

{
  "image_quality": {
    "grade": "GOOD | ACCEPTABLE | POOR",
    "issues": [],
    "confidence": "HIGH | MODERATE | LIMITED"
  },
  "basic_parameters": {
    "heart_rate": "",
    "rhythm": "",
    "pr_interval_ms": "",
    "qrs_duration_ms": "",
    "qt_interval_ms": "",
    "qtc_ms": "",
    "qtc_status": "NORMAL | BORDERLINE | PROLONGED | CRITICALLY PROLONGED"
  },
  "findings": {
    "p_wave": "",
    "axis": "",
    "qrs_complex": "",
    "st_segment": "",
    "t_wave": "",
    "u_wave": "",
    "rhythm_details": ""
  },
  "chamber_analysis": {
    "lvh": false,
    "rvh": false,
    "lae": false,
    "rae": false,
    "details": ""
  },
  "primary_diagnosis": "",
  "secondary_findings": [],
  "critical_alerts": [],
  "india_specific_considerations": [],
  "age_risk_analysis": "",
  "differential_diagnosis": [],
  "recommended_actions": {
    "immediate": [],
    "urgent": [],
    "routine": [],
    "additional_investigations": []
  },
  "medications_to_flag": [],
  "follow_up_ecg_needed": true,
  "cardiologist_review_priority": "STAT | URGENT | ROUTINE",
  "summary_for_doctor": "",
  "patient_friendly_summary": ""
}

---

CRITICAL SAFETY RULES:
1. ALWAYS output "cardiologist_review_priority": "STAT" for any STEMI pattern, 
   malignant arrhythmia, critical QTc >500ms, complete heart block, or Brugada Type 1
2. NEVER downplay a finding to be reassuring — err on the side of caution
3. If image quality is poor, state clearly what cannot be assessed and what can
4. Always contextualalize with age — a "normal variant" in a 25-year-old is a red 
   flag in a 55-year-old with hypertension
5. This output is for DOCTOR review — it is a clinical decision support tool, 
   not a replacement for physician judgment
```

---

## 🇮🇳 SECTION 2 — India-Specific Clinical Context (Critical Knowledge)

### 2.1 Epidemiology Claude Must Know

| Condition | India Reality vs. Western Norms |
|---|---|
| **CAD onset age** | Indians develop CAD 10 years earlier than Western counterparts |
| **Premature MI** | MI in men <40 years is significantly more common in India |
| **Rheumatic Heart Disease** | Still highly prevalent — causes mitral stenosis, AF, TR in 20–50 yr age group |
| **Hypertension** | ~30% urban Indians, often poorly controlled, leading LVH cause |
| **Diabetes** | India is the diabetes capital — silent ischemia is rampant; ECG must be interpreted knowing diabetics have painless MI |
| **LP(a)** | Genetically elevated in South Asians — major premature CAD driver |
| **Low HDL** | Common in Indians even without obesity — metabolic risk |
| **Thin-fat Indians** | Normal BMI but high visceral fat — metabolic syndrome underdiagnosed |
| **COPD** | High prevalence from air pollution, biomass cooking fuel — RV strain, P pulmonale |
| **Tuberculosis** | Pericarditis, myocarditis, conduction disease — active or sequelae |
| **Anemia** | Sinus tachycardia, high-output state — very common in Indian women |
| **Hypothyroidism** | Bradycardia, low voltage, long QT — common in Indian women |
| **Alcohol** | Binge pattern in some populations → AF, cardiomyopathy, QTc prolongation |

### 2.2 Indian Genetic Risk Factors to Flag

- **High LP(a)**: Premature atherosclerosis, not detected by standard lipid panel
- **APO E4 variant**: Higher CAD risk
- **PCSK9 mutations**: Familial hypercholesterolemia underdiagnosed in India
- **HCM**: MYBPC3 gene deletion (25bp) — founder mutation in South Asians — most common genetic cardiac condition in Indians, presents as unexplained LVH on ECG
- **Long QT syndrome variants**: Congenital LQTS underdiagnosed in India
- **Brugada Syndrome**: Emerging data shows it is not rare in India

### 2.3 Common Indian Medications That Affect ECG

| Medication | ECG Effect | Prevalence |
|---|---|---|
| Digoxin | Scooped ST, AV block | Widely used for AF, heart failure |
| Hydroxychloroquine | QTc prolongation, conduction block | Used for RA, lupus, historically COVID |
| Azithromycin | QTc prolongation | Massively over-prescribed in India |
| Fluoroquinolones | QTc prolongation | Very commonly used |
| Tricyclic antidepressants | QTc, wide QRS, right axis | Used in psychiatry and pain |
| Antifungals (azoles) | QTc prolongation | Common for fungal infections |
| Lithium | T wave changes, sinus node dysfunction | Psychiatric patients |
| Sotalol / Amiodarone | QTc, bradycardia, conduction changes | Arrhythmia management |
| Herbal / Ayurvedic | Aconite poisoning → severe bradycardia, VT | Underreported |

---

## 👤 SECTION 3 — Patient Context Input Schema

Every API call must inject a structured patient context. This is the **runtime prompt injection** that personalizes the ECG interpretation. Build this from the doctor's intake form.

### 3.1 Patient Context JSON Schema

```json
{
  "patient_context": {
    
    "demographics": {
      "age": 52,
      "sex": "male",
      "state_region": "Tamil Nadu",
      "urban_rural": "urban",
      "occupation": "software engineer (sedentary)",
      "language": "Tamil"
    },

    "presenting_complaint": {
      "chief_complaint": "chest pain",
      "duration": "2 hours",
      "character": "crushing, radiating to left arm",
      "onset": "sudden",
      "associated_symptoms": ["sweating", "nausea"],
      "exertional_or_rest": "rest"
    },

    "vitals_at_presentation": {
      "bp_systolic": 160,
      "bp_diastolic": 100,
      "heart_rate": 88,
      "spo2_percent": 97,
      "temperature_celsius": 37.2,
      "respiratory_rate": 18
    },

    "cardiac_history": {
      "prior_mi": false,
      "prior_mi_year": null,
      "prior_pci_cabg": false,
      "known_cad": false,
      "heart_failure": false,
      "prior_arrhythmia": false,
      "rheumatic_fever_history": false,
      "known_valve_disease": false,
      "congenital_heart_disease": false,
      "prior_ecg_available": false,
      "icd_pacemaker": false
    },

    "medical_history": {
      "hypertension": true,
      "hypertension_years": 8,
      "diabetes": true,
      "diabetes_type": "Type 2",
      "diabetes_years": 5,
      "dyslipidemia": true,
      "ckd": false,
      "ckd_stage": null,
      "copd_asthma": false,
      "thyroid_disorder": false,
      "thyroid_type": null,
      "stroke_tia": false,
      "peripheral_arterial_disease": false,
      "autoimmune_disease": false,
      "tuberculosis_history": false,
      "tb_treatment_completed": null,
      "hiv": false,
      "anemia": false,
      "liver_disease": false,
      "obesity": true,
      "bmi": 28.4,
      "obstructive_sleep_apnea": false
    },

    "family_history": {
      "premature_cad": true,
      "premature_cad_details": "Father had MI at age 48",
      "sudden_cardiac_death": false,
      "hcm_or_cardiomyopathy": false,
      "familial_hypercholesterolemia": false,
      "long_qt_or_channelopathy": false,
      "rheumatic_heart_disease": false,
      "hypertension": true,
      "diabetes": true
    },

    "lifestyle": {
      "smoking_status": "ex-smoker",
      "smoking_pack_years": 15,
      "smoking_quit_year": 2019,
      "alcohol": "occasional",
      "alcohol_units_per_week": 4,
      "tobacco_chewing": false,
      "physical_activity": "sedentary",
      "diet_type": "non-vegetarian",
      "diet_pattern": "high refined carbohydrate, oily food",
      "stress_level": "high",
      "sleep_hours": 5,
      "betel_nut_use": false
    },

    "current_medications": [
      { "drug": "Metformin", "dose": "1000mg", "frequency": "BD" },
      { "drug": "Amlodipine", "dose": "5mg", "frequency": "OD" },
      { "drug": "Atorvastatin", "dose": "40mg", "frequency": "HS" },
      { "drug": "Aspirin", "dose": "75mg", "frequency": "OD" }
    ],

    "recent_events": {
      "recent_illness": false,
      "recent_viral_fever": false,
      "recent_surgery": false,
      "recent_hospitalization": false,
      "recent_long_travel": false,
      "recent_immobilization": false,
      "covid_history": true,
      "covid_year": 2022
    },

    "lab_values_if_available": {
      "hemoglobin_g_dl": 13.2,
      "serum_potassium_meq_l": 4.1,
      "serum_sodium_meq_l": 138,
      "creatinine_mg_dl": 1.1,
      "tsh_uiu_ml": null,
      "hba1c_percent": 8.2,
      "total_cholesterol_mg_dl": 210,
      "ldl_mg_dl": 135,
      "hdl_mg_dl": 38,
      "triglycerides_mg_dl": 280,
      "troponin_i_ng_ml": null,
      "bnp_or_nt_pro_bnp": null,
      "d_dimer": null,
      "magnesium": null
    },

    "ecg_context": {
      "ecg_indication": "chest pain workup",
      "time_of_ecg": "14:30",
      "serial_ecg_number": 1,
      "is_serial_comparison": false
    }
  }
}
```

---

## 🔬 SECTION 4 — Complete ECG Findings Matrix

### What Claude Should Identify & Analyze Per Lead

| Lead Group | What to Analyze | Clinical Significance |
|---|---|---|
| **Lead I** | Axis, lateral wall | Lateral MI, LAD |
| **Lead II** | Inferior wall, P wave, rhythm strip | Inferior MI, P morphology |
| **Lead III** | Inferior wall, axis | Inferior MI (SIII+QIII = PE) |
| **aVR** | Global subendocardial ischemia, LMCA | ST elevation aVR = LMCA/proximal LAD occlusion — CRITICAL |
| **aVL** | High lateral wall | High lateral MI |
| **aVF** | Inferior wall | Inferior MI |
| **V1** | Septal, RV, RBBB, Brugada, posterior MI | Broad R = posterior MI or RBBB |
| **V2** | Septal, anterior, Wellens | Wellens T inversions = LAD stenosis |
| **V3** | Anterior transition | Poor R progression |
| **V4** | Anterior wall | Anterior MI territory |
| **V5** | Lateral, LVH voltage | Lateral MI |
| **V6** | Lateral, LVH voltage | Lateral MI |

### 4.1 Complete Pathology Identification Checklist

**STEMI Patterns:**
- [ ] Inferior STEMI (II, III, aVF) + RV check (V3R, V4R)
- [ ] Anterior STEMI (V1–V4)
- [ ] Anteroseptal STEMI (V1–V3)
- [ ] Extensive Anterior STEMI (V1–V6)
- [ ] Lateral STEMI (I, aVL, V5, V6)
- [ ] Posterior STEMI (ST depression V1–V3, tall R V1)
- [ ] High Lateral STEMI (I, aVL)
- [ ] STEMI equivalent — De Winter pattern (V1–V6 upsloping ST depression + peaked T)
- [ ] STEMI equivalent — Wellens Syndrome Type A (biphasic T V2–V3) and Type B (deep symmetric T)
- [ ] STEMI equivalent — aVR ST elevation > V1 (LMCA)

**NSTEMI/UA Patterns:**
- [ ] Horizontal ST depression ≥1mm in 2+ contiguous leads
- [ ] Downsloping ST depression
- [ ] Dynamic T wave changes
- [ ] Symmetric T wave inversions (anterior, inferior, lateral)

**Arrhythmias:**
- [ ] Sinus tachycardia / bradycardia / arrhythmia
- [ ] Atrial fibrillation (absent P, irregular R-R, fibrillatory baseline)
- [ ] Atrial flutter (sawtooth, rate 300 with 2:1 or 4:1 block)
- [ ] Supraventricular tachycardia (AVNRT, AVRT, AT)
- [ ] Wolff-Parkinson-White (short PR, delta wave, wide QRS)
- [ ] Premature Atrial Complexes
- [ ] Premature Ventricular Complexes — unifocal/multifocal, bigeminy/trigeminy
- [ ] Ventricular Tachycardia — monomorphic/polymorphic (Torsades)
- [ ] Ventricular Fibrillation
- [ ] Idioventricular rhythm
- [ ] Junctional rhythm
- [ ] 1st Degree AV Block (PR >200ms)
- [ ] 2nd Degree AV Block Mobitz I (progressive PR lengthening)
- [ ] 2nd Degree AV Block Mobitz II (fixed PR, dropped beats)
- [ ] 2:1 AV Block (need serial ECGs to differentiate Mobitz I vs II)
- [ ] Complete (3rd degree) AV Block (dissociated P and QRS)
- [ ] RBBB (rSR' in V1, wide S in I, V5, V6)
- [ ] LBBB (broad notched R in I, V5, V6, no septal Q)
- [ ] Incomplete RBBB / LBBB
- [ ] Left Anterior Fascicular Block (LAD, small Q in I, small R in III)
- [ ] Left Posterior Fascicular Block (RAD, small R in I, small Q in III)
- [ ] Bifascicular block (RBBB + LAFB or LPFB)
- [ ] Trifascicular block (bifascicular + PR prolongation)

**Special Syndromes:**
- [ ] Brugada Type 1 (coved ST elevation ≥2mm V1/V2)
- [ ] Brugada Type 2/3 (saddle-shape — needs provocation)
- [ ] Long QT Syndrome (QTc >500ms, T wave morphology)
- [ ] Short QT Syndrome (QTc <340ms)
- [ ] HCM Pattern (deep septal Q waves, massive voltage, LVH)
- [ ] ARVC (epsilon wave, right precordial T inversions, LBBB VT)
- [ ] Pericarditis (diffuse concave ST elevation, PR depression)
- [ ] Myocarditis (diffuse changes, any arrhythmia)
- [ ] Pulmonary Embolism (S1Q3T3, sinus tach, new RBBB, RV strain)
- [ ] Hyperkalemia (peaked T, wide QRS, sine wave)
- [ ] Hypokalemia (prominent U waves, flat T, QTU prolongation)
- [ ] Hypercalcemia (short QT, short ST)
- [ ] Hypocalcemia (long QT, long ST)
- [ ] Hypothyroidism (low voltage, bradycardia, flat T)
- [ ] Digitalis toxicity (scooped ST, AV block, bidirectional VT)
- [ ] Takotsubo (deep T inversions V1–V5, QTc prolongation post-stress)
- [ ] Early Repolarization (benign vs. malignant — inferior/lateral distribution)

---

## 🔁 SECTION 5 — Age-Stratified Risk Framework (India-Specific)

### Age Group Specific Interpretive Guidelines

| Age Group | ECG Context for India |
|---|---|
| **<30 years** | Congenital channelopathies (LQTS, Brugada, WPW), HCM, myocarditis, rheumatic heart disease, drug toxicity |
| **30–45 years** | Premature CAD (especially men — Indian male epidemic), rheumatic AF, viral cardiomyopathy, cocaine/substance in urban populations |
| **45–60 years** | Peak CAD presentation in Indian men. Hypertensive LVH, diabetic silent ischemia, metabolic syndrome. Women: pre-menopausal protection wanes |
| **60–70 years** | Multi-vessel CAD, AF, heart failure, conduction disease, valve disease (degenerative + rheumatic), CKD-related changes |
| **>70 years** | Sick sinus syndrome, complete heart block, severe valve disease, multi-morbidity. Polypharmacy ECG effects prominent |

### Age-Based QTc Warning Thresholds

| Sex | Borderline | Prolonged | Critical |
|---|---|---|---|
| Male | 430–440ms | 441–500ms | >500ms |
| Female | 450–460ms | 461–500ms | >500ms |

---

## 💉 SECTION 6 — API Integration Blueprint

### 6.1 Full API Call Structure

```javascript
const interpretECG = async (ecgImageBase64, patientContext) => {
  
  const systemPrompt = `[Insert full Master System Prompt from Section 1]`;
  
  const userMessage = {
    role: "user",
    content: [
      {
        type: "image",
        source: {
          type: "base64",
          media_type: "image/jpeg",
          data: ecgImageBase64
        }
      },
      {
        type: "text",
        text: `
Please perform a complete ECG interpretation for this patient.

## PATIENT CONTEXT:
${JSON.stringify(patientContext, null, 2)}

## IMPORTANT INDIA-SPECIFIC FLAGS FOR THIS PATIENT:
- Age ${patientContext.patient_context.demographics.age} — ${getAgeRiskNote(patientContext)}
- Key risk factors: ${getRiskSummary(patientContext)}
- Medications to watch for ECG effects: ${getMedFlags(patientContext)}
- Genetic risks: ${getGeneticFlags(patientContext)}
- Regional context: ${patientContext.patient_context.demographics.state_region}

## YOUR TASK:
Interpret this 12-lead ECG with the clinical gravitas of a senior cardiologist at a 
top-tier Indian cardiac center. Account for ALL patient factors above. 
Output your interpretation in the EXACT JSON format specified in your instructions.

Remember: 
1. You are the last line of clinical intelligence before a human cardiologist reviews
2. Indian patients present differently — premature CAD is common, rheumatic disease 
   is real, and "normal for age" thresholds from Western literature may not apply
3. The doctor reading your output will use it to make immediate clinical decisions
`
      }
    ]
  };

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "anthropic-beta": "interleaved-thinking-2025-05-14"
      // Note: API key is injected by the backend — never expose in frontend
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4000,
      thinking: {
        type: "enabled",
        budget_tokens: 2000  // Enable extended thinking for complex ECGs
      },
      system: systemPrompt,
      messages: [userMessage]
    })
  });

  const data = await response.json();
  
  // Extract text from content blocks (thinking + text mixed)
  const textContent = data.content
    .filter(block => block.type === "text")
    .map(block => block.text)
    .join("\n");
  
  // Parse JSON from response
  const jsonMatch = textContent.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    return JSON.parse(jsonMatch[0]);
  }
  
  return { error: "Could not parse structured response", raw: textContent };
};
```

### 6.2 Helper Functions for Dynamic Prompt Injection

```javascript
const getAgeRiskNote = (ctx) => {
  const age = ctx.patient_context.demographics.age;
  const sex = ctx.patient_context.demographics.sex;
  
  if (age < 30) return "Young patient — high suspicion for channelopathy, HCM, myocarditis, rheumatic disease";
  if (age < 45 && sex === "male") return "PREMATURE CAD RISK — Indian male in peak premature MI age window";
  if (age < 45 && sex === "female") return "Consider rheumatic heart disease, cardiomyopathy, autoimmune cardiac involvement";
  if (age < 60) return "Peak CAD age for Indian population — interpret ischemic changes with high clinical concern";
  if (age >= 60) return "Multi-morbidity expected — check for polypharmacy effects, conduction disease, valve disease";
  return "Standard adult risk";
};

const getRiskSummary = (ctx) => {
  const h = ctx.patient_context.medical_history;
  const r = [];
  if (h.hypertension) r.push(`HTN x${h.hypertension_years}y`);
  if (h.diabetes) r.push(`DM-${h.diabetes_type} x${h.diabetes_years}y (HbA1c ${ctx.patient_context.lab_values_if_available?.hba1c_percent}%)`);
  if (h.dyslipidemia) r.push("Dyslipidemia");
  if (h.copd_asthma) r.push("COPD — RV strain risk");
  if (h.ckd) r.push(`CKD Stage ${h.ckd_stage} — electrolyte/QTc risk`);
  if (ctx.patient_context.lifestyle.smoking_status === "current") r.push("Active smoker");
  if (ctx.patient_context.lifestyle.smoking_status === "ex-smoker") r.push(`Ex-smoker ${ctx.patient_context.lifestyle.smoking_pack_years} pack-years`);
  if (ctx.patient_context.family_history.premature_cad) r.push("FAMILY: Premature CAD");
  return r.join(", ") || "No major risk factors documented";
};

const getMedFlags = (ctx) => {
  const qtProlongingDrugs = [
    "azithromycin", "ciprofloxacin", "levofloxacin", "metronidazole",
    "hydroxychloroquine", "amiodarone", "sotalol", "haloperidol",
    "lithium", "tricyclic", "antifungal", "fluconazole", "itraconazole",
    "ondansetron", "domperidone"
  ];
  
  const meds = ctx.patient_context.current_medications || [];
  const flags = meds
    .filter(m => qtProlongingDrugs.some(d => m.drug.toLowerCase().includes(d)))
    .map(m => `${m.drug} (QTc risk)`);
    
  return flags.length > 0 ? flags.join(", ") : "None identified";
};

const getGeneticFlags = (ctx) => {
  const fh = ctx.patient_context.family_history;
  const flags = [];
  if (fh.premature_cad) flags.push("Familial CAD — consider high LP(a), FH");
  if (fh.hcm_or_cardiomyopathy) flags.push("HCM risk — MYBPC3 Indian founder mutation");
  if (fh.long_qt_or_channelopathy) flags.push("Channelopathy risk — assess QTc, Brugada carefully");
  if (fh.sudden_cardiac_death) flags.push("FAMILY SCD — high-risk genetic arrhythmia possible");
  if (fh.familial_hypercholesterolemia) flags.push("FH family history — premature atherosclerosis risk");
  return flags.join("; ") || "No significant genetic flags";
};
```

---

## 🖥️ SECTION 7 — Output Display Framework

### 7.1 Structured Report Sections to Render in UI

```
┌─────────────────────────────────────────────────────┐
│  🚨 CRITICAL ALERTS (if any)                        │
│  [Red banner — immediate attention]                  │
├─────────────────────────────────────────────────────┤
│  BASIC PARAMETERS                                    │
│  HR | Rhythm | PR | QRS | QTc | Axis                │
├─────────────────────────────────────────────────────┤
│  PRIMARY DIAGNOSIS                                   │
│  [Bold, prominent]                                   │
├─────────────────────────────────────────────────────┤
│  DETAILED FINDINGS                                   │
│  P Wave | QRS | ST | T Wave | Rhythm                 │
├─────────────────────────────────────────────────────┤
│  INDIA-SPECIFIC CONSIDERATIONS                       │
│  [Highlighted section]                               │
├─────────────────────────────────────────────────────┤
│  DIFFERENTIAL DIAGNOSIS                              │
│  [Ranked list]                                       │
├─────────────────────────────────────────────────────┤
│  RECOMMENDED ACTIONS                                 │
│  Immediate | Urgent | Routine | Investigations       │
├─────────────────────────────────────────────────────┤
│  REVIEW PRIORITY: [STAT | URGENT | ROUTINE]          │
│  ⚠️ For Cardiologist Review                          │
└─────────────────────────────────────────────────────┘
```

---

## ✅ SECTION 8 — Quality Assurance Checklist

Before deploying this system in a clinical setting, ensure:

- [ ] API key is stored server-side only — never in frontend code
- [ ] All ECG outputs are logged with timestamp, patient ID (anonymized), and AI version
- [ ] Mandatory disclaimer displayed: *"AI-assisted interpretation — requires cardiologist verification"*
- [ ] STAT alerts trigger SMS/notification to on-call cardiologist
- [ ] Rejection flow for unreadable images with clear user instruction
- [ ] All patient data encrypted in transit (TLS 1.3) and at rest (AES-256)
- [ ] DISHA / IT Act 2000 compliance for Indian patient health data
- [ ] Audit trail for every ECG interpretation stored for minimum 7 years (Indian medical records law)
- [ ] The system must NEVER be marketed as a diagnostic device without CDSCO approval

---

## ⚖️ SECTION 9 — Regulatory & Ethical Framework (India-Specific)

| Requirement | Detail |
|---|---|
| **CDSCO Approval** | AI-based medical software may require classification under Medical Devices Rules 2017 |
| **DISHA Compliance** | Digital Information Security in Healthcare Act — protect all patient health data |
| **IT Act 2000** | Data privacy and cybersecurity obligations |
| **Medical Council** | MCI guidelines on telemedicine and AI-assisted diagnosis (2020 guidelines apply) |
| **Liability** | The treating physician retains full clinical and legal responsibility |
| **Consent** | Inform patients that AI tools assist interpretation — document in record |
| **Disclaimer** | Every report must state: "This is a clinical decision support output. Validated interpretation by a qualified cardiologist is required before clinical action." |

---

## 🚀 SECTION 10 — Next Steps for Development

1. **Phase 1**: Build the core image upload + Claude API integration + structured JSON report UI
2. **Phase 2**: Add patient context intake form (the schema in Section 3)
3. **Phase 3**: Serial ECG comparison (send 2 ECGs, ask Claude to compare)
4. **Phase 4**: STAT alert system — push notification to cardiologist on STEMI detection
5. **Phase 5**: Regional language output (Tamil, Hindi, Telugu, Malayalam) for patient summaries
6. **Phase 6**: CDSCO regulatory submission preparation
7. **Phase 7**: Prospective validation study at partner hospital — compare AI output vs. cardiologist gold standard, target AUC >0.92

---

*This document was prepared for a clinical AI product targeting the Indian healthcare ecosystem. All prompt engineering is designed to leverage Claude Sonnet's medical reasoning capabilities, calibrated specifically for Indian epidemiology, genetics, and clinical context.*

**Last Updated**: 2025 | **Powered by**: Anthropic Claude Sonnet Vision API
