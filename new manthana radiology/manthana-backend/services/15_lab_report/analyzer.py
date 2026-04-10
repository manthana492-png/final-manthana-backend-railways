"""
Manthana — Lab Report Analyzer (V2 with Google MedGemma-4B-IT)

Architecture:
1. Structured parsing (shared/medical_document_parser → google/medgemma-4b-it) → PDF/image/text
2. Clinical interpretation (OpenRouter; SSOT config/cloud_inference.yaml role lab_report)
3. Correlation-ready output → labs/structured fields for correlation_engine

Benefits:
- Better accuracy on scanned documents (vision + multimodal model)
- Structured data enables correlation rules to fire properly
- Single cloud LLM path (OpenRouter)
- Handles both text PDFs and image-based reports
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Optional

from critical_values import check_critical_values, normalize_labs_for_critical

logger = logging.getLogger("manthana.lab_analyzer")


def _shared_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "shared"


def _ensure_shared_on_path() -> None:
    for p in (Path("/app/shared"), _shared_path()):
        s = str(p.resolve())
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


def parse_clinical_notes(raw: str) -> dict:
    """Parse 'key:val; key:val' string from gateway into structured dict."""
    ctx = {}
    if not raw or not raw.strip():
        return ctx
    for part in raw.split(";"):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            ctx[k.strip().lower().replace(" ", "_")] = v.strip()
    return ctx


def _finalize_lab_output(output: dict) -> dict:
    """Normalize labs for correlation/critical checks; populate critical_values list."""
    labs_in = output.get("labs")
    if labs_in is None:
        labs_in = {}
    if not isinstance(labs_in, dict):
        labs_in = {}
    norm = normalize_labs_for_critical(labs_in)
    output["labs"] = norm
    output["critical_values"] = check_critical_values(norm)
    return output

# ═════════════════════════════════════════════════════════════════════════════
# Import shared modules (with graceful fallback)
# ═════════════════════════════════════════════════════════════════════════════

try:
    import sys
    sys.path.insert(0, "/app/shared")
    from medical_document_parser import parse_lab_report as _medgemma_parse

    MEDGEMMA_PARSER_AVAILABLE = True
except ImportError as e:
    logger.warning("MedGemma medical_document_parser not available: %s", e)
    MEDGEMMA_PARSER_AVAILABLE = False
    _medgemma_parse = None

# Back-compat for any code checking the old name
PARROTV_AVAILABLE = MEDGEMMA_PARSER_AVAILABLE

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

# USE_MEDGEMMA_PARSER preferred; USE_PARROTV_PARSER still honored (legacy).
USE_STRUCTURED_PARSER = (
    os.getenv("USE_MEDGEMMA_PARSER") or os.getenv("USE_PARROTV_PARSER", "auto")
).lower()

MEDGEMMA_PARSER_MODEL_ID = "google/medgemma-4b-it"
# Options: "always" | "never" | "auto"
# "auto" = use if available, fallback to text-only if not


# ═════════════════════════════════════════════════════════════════════════════
# Text Extraction (Legacy - for fallback)
# ═════════════════════════════════════════════════════════════════════════════

def extract_text(filepath: str) -> str:
    """Extract text from PDF, TXT, CSV, or image."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _extract_pdf_text(filepath)
    elif ext in (".txt", ".csv", ".tsv", ".text"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        if MEDGEMMA_PARSER_AVAILABLE and USE_STRUCTURED_PARSER != "never":
            return "[IMAGE_USE_PARSER]"  # Signal to use vision parser
        else:
            return _extract_image_text_fallback(filepath)
    else:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf_text(filepath: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz
    text_parts = []
    with fitz.open(filepath) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    text = "\n".join(text_parts).strip()
    return text


def _extract_image_text_fallback(filepath: str) -> str:
    """Fallback when image parsing not available."""
    raise ValueError(
        "Image-based lab reports require the MedGemma medical document parser "
        f"({MEDGEMMA_PARSER_MODEL_ID}). Enable it or upload a text-based PDF."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Core Analysis Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def analyze_lab_report(
    filepath: str,
    use_parser: Optional[str] = None,
    patient_context: Optional[dict] = None,
) -> dict:
    """
    Analyze a lab report file using structured parsing + OpenRouter.
    
    Args:
        filepath: Path to the lab report file
        use_parser: Override parser usage ("yes", "no", or None for env default)
        patient_context: Optional patient info (age, gender) for better interpretation
    
    Returns:
        AnalysisResponse-compatible dict with:
        - findings: List of abnormal values with severity
        - impression: Clinical summary
        - pathology_scores: Normalized scores
        - structured: Full structured data from parser
        - labs: Flattened key values for correlation_engine
    """
    # Determine parser usage
    should_use_parser = _should_use_parser(filepath, use_parser)
    
    # Step 1: Structured parsing (if enabled and available)
    structured_result = None
    if should_use_parser and MEDGEMMA_PARSER_AVAILABLE and _medgemma_parse:
        try:
            logger.info("Using MedGemma parser for: %s", filepath)
            structured_result = _medgemma_parse(filepath)
            logger.info(
                "Parser success: %s",
                structured_result.get("document_type", "unknown"),
            )
        except Exception as e:
            logger.warning("MedGemma parser failed: %s. Falling back to text extraction.", e)
            structured_result = None
    
    # Step 2: Text extraction (fallback or supplementary)
    raw_text = ""
    if structured_result and structured_result.get("raw_text"):
        raw_text = structured_result["raw_text"]
    else:
        try:
            raw_text = extract_text(filepath)
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            raw_text = ""

    # Step 2b: Image uploads emit a parser token when vision path is required
    if raw_text.strip() == "[IMAGE_USE_PARSER]":
        if MEDGEMMA_PARSER_AVAILABLE and _medgemma_parse:
            try:
                structured_result = _medgemma_parse(filepath)
            except Exception as e:
                logger.warning("MedGemma parse after image token failed: %s", e)
                structured_result = None
        else:
            structured_result = None
        if structured_result and structured_result.get("structured"):
            interpretation = _interpret_with_structured(
                structured=structured_result["structured"],
                raw_text=structured_result.get("raw_text") or "",
                patient_context=patient_context,
            )
            output = _merge_outputs(
                structured_result,
                interpretation,
                structured_result.get("raw_text") or "",
            )
            return _finalize_lab_output(output)
        if structured_result and structured_result.get("raw_text"):
            raw_text = structured_result["raw_text"]
        else:
            interpretation = {
                "findings": [
                    {
                        "label": "Image Not Parseable",
                        "severity": "warning",
                        "confidence": 100.0,
                        "region": "Laboratory report",
                        "description": (
                            "Could not extract text or structure from the image. "
                            "Upload a clearer photo or a text-based PDF."
                        ),
                    }
                ],
                "impression": "Lab report image could not be processed automatically.",
                "pathology_scores": {},
                "structures": [],
                "detected_region": "lab_report",
            }
            output = _merge_outputs(None, interpretation, "")
            return _finalize_lab_output(output)

    # Step 3: Clinical interpretation via OpenRouter
    if structured_result and structured_result.get("structured"):
        interpretation = _interpret_with_structured(
            structured=structured_result["structured"],
            raw_text=raw_text,
            patient_context=patient_context,
        )
    else:
        interpretation = _interpret_raw_text(raw_text, patient_context)

    # Step 4: Merge and format output
    output = _merge_outputs(structured_result, interpretation, raw_text)
    return _finalize_lab_output(output)


def _should_use_parser(filepath: str, override: Optional[str]) -> bool:
    """Determine if we should use the structured parser."""
    if override == "yes":
        return True
    if override == "no":
        return False
    
    # Check env setting
    if USE_STRUCTURED_PARSER == "never":
        return False
    if USE_STRUCTURED_PARSER == "always":
        return MEDGEMMA_PARSER_AVAILABLE
    
    # Auto: use for images, optional for PDFs with sparse text
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        return MEDGEMMA_PARSER_AVAILABLE  # Required for images
    
    # For PDFs, check if text extraction yields sparse results
    if ext == ".pdf":
        try:
            text = _extract_pdf_text(filepath)
            # If very little text, likely a scanned PDF
            if len(text.strip()) < 100:
                logger.info("PDF appears scanned (sparse text). Will use vision parser.")
                return MEDGEMMA_PARSER_AVAILABLE
        except Exception:
            pass
    
    # Default: use if available for better structured data
    return MEDGEMMA_PARSER_AVAILABLE


_INDIA_SYSTEM_STRUCTURED = """\
You are a senior clinical pathologist at a leading tertiary-care hospital in India (NABL/CAP accredited).
You have deep expertise in Indian disease burden and epidemiology:
- Nutritional anaemia (iron-deficiency, B12, folate) is extremely common — especially in women and children.
- Haemoglobin cut-offs: Men <13 g/dL, Women <12 g/dL, Children <11 g/dL (WHO India).
- Vitamin D deficiency is endemic (>70% of Indians are deficient; <20 ng/mL is deficient).
- Type-2 Diabetes and pre-diabetes: HbA1c ≥6.5% = diabetic; 5.7–6.4% = pre-diabetic (IDF/ADA).
- TB (tuberculosis) is highly prevalent — always consider when ESR ↑↑, lymphocytosis, Hb↓, weight loss.
- Dengue: Platelet count <100,000/µL warrants urgent monitoring; <50,000/µL is critical.
- HBsAg positivity: ~2–4% of Indian adults; correlate with LFT (SGPT/SGOT).
- Thyroid dysfunction (hypothyroidism) is underdiagnosed, especially in women — TSH >4.5 mIU/L.
- CKD from diabetic nephropathy is rising — Creatinine + eGFR + urine albumin are key.
- Indian lab abbreviations: Hb, TLC, DLC, ESR, Sr.Creatinine, Sr.Uric Acid, SGPT/ALT, SGOT/AST,
  LFT, RFT, TSH, FT3, FT4, HbA1c, FBS, PPBS, Sr.Cholesterol, TG, LDL, HDL, VLDL, HBsAg, Anti-HCV.
- Reference ranges may vary slightly by Indian lab; use the lab's printed range if available.

Your task: interpret structured lab data, flag ALL abnormals, provide India-specific clinical context,
and suggest practical follow-up for an Indian patient. ALWAYS respond with ONLY valid JSON.\
"""

_INDIA_JSON_SCHEMA = """\
{
  "findings": [
    {
      "label": "Short name e.g. 'Low Haemoglobin (Anaemia)'",
      "severity": "critical|warning|info|clear",
      "confidence": 0-100,
      "region": "test parameter name",
      "description": "Clinical explanation with India context (e.g. likely iron-deficiency given endemic prevalence)"
    }
  ],
  "impression": "1–2 paragraph India-contextualised clinical summary",
  "pathology_scores": {"parameter_name": 0.0-1.0},
  "follow_up_recommendations": [
    "Specific, actionable India-relevant recommendation e.g. 'Start iron supplementation (IFA) as per NIN guidelines'"
  ],
  "india_risk_flags": {
    "tb_risk": "low|moderate|high",
    "diabetes_status": "normal|pre-diabetic|diabetic|unknown",
    "anaemia_type": "iron-deficiency|B12|folate|mixed|haemolytic|unknown|none",
    "dengue_alert": true
  },
  "detected_region": "lab_report"
}\
"""


def _interpret_with_structured(
    structured: dict,
    raw_text: str,
    patient_context: Optional[dict],
) -> dict:
    """Interpret structured parser output with India-focused clinical analysis (OpenRouter)."""
    structured_json = json.dumps(structured, indent=2, default=str)[:8000]

    user_prompt = (
        f"Analyze this structured Indian lab report data:\n\n"
        f"STRUCTURED DATA:\n{structured_json}\n\n"
        f"RAW TEXT (for reference):\n{raw_text[:3000]}\n"
    )
    if patient_context:
        user_prompt += f"\nPATIENT CONTEXT:\n{json.dumps(patient_context, indent=2)}\n"
    user_prompt += f"\nRespond ONLY with valid JSON matching this schema:\n{_INDIA_JSON_SCHEMA}"

    return _call_openrouter_lab(_INDIA_SYSTEM_STRUCTURED, user_prompt)


_INDIA_SYSTEM_TEXT = """\
You are a senior clinical pathologist at a leading NABL-accredited hospital in India.
You are highly familiar with:
- Indian disease burden: iron-deficiency anaemia, Vitamin D deficiency, hypothyroidism,
  Type-2 Diabetes (HbA1c), TB, dengue (low platelets), HBsAg positivity, CKD.
- Indian lab report formats: printed as tabular text from software like SRL, Thyrocare,
  Dr. Lal PathLabs, Metropolis, NABL labs. Values may have H/L/HH/LL flags.
- Indian population reference ranges (Hb: Men ≥13, Women ≥12, Children ≥11 g/dL).
- Interpret borderline values in Indian epidemiological context — e.g. Hb 11.5 g/dL in
  an Indian woman of reproductive age is likely iron-deficiency until proven otherwise.

Your task: extract ALL test values from the raw text, identify abnormals, and provide
India-specific clinical interpretation. Respond ONLY with valid JSON.\
"""


def _interpret_raw_text(raw_text: str, patient_context: Optional[dict]) -> dict:
    """Interpret raw lab report text with India-focused clinical analysis (OpenRouter)."""
    user_prompt = (
        f"Analyze this Indian lab report text and extract all values, flag abnormals, "
        f"and provide India-specific clinical interpretation:\n\n"
        f"{raw_text[:12000]}"
    )
    if patient_context:
        user_prompt += f"\n\nPatient context: {json.dumps(patient_context)}"
    user_prompt += f"\n\nRespond ONLY with valid JSON matching this schema:\n{_INDIA_JSON_SCHEMA}"

    return _call_openrouter_lab(_INDIA_SYSTEM_TEXT, user_prompt)


def _call_openrouter_lab(system_prompt: str, user_prompt: str) -> dict:
    """Clinical JSON interpretation via OpenRouter (role lab_report)."""
    _ensure_shared_on_path()
    try:
        from llm_router import llm_router
    except Exception as e:
        logger.warning("llm_router import failed: %s", e)
        return _fallback_response("LLM router not available")

    try:
        out = llm_router.complete_for_role(
            "lab_report",
            system_prompt,
            user_prompt,
            temperature=0.1,
            max_tokens=4096,
            requires_json=True,
        )
    except ValueError as e:
        return _fallback_response(str(e))
    except Exception as e:
        logger.error("OpenRouter lab interpretation error: %s", e)
        return _fallback_response(str(e))

    raw = (out.get("content") or "").strip()
    parsed = _parse_llm_response(raw)
    mu = str(out.get("model_used") or "").strip()
    parsed["_llm_model_used"] = f"openrouter:{mu}" if mu else "openrouter:lab_report"
    return parsed


def _parse_llm_response(raw: str) -> dict:
    """Parse JSON from LLM response."""
    # Try to extract JSON from markdown fences
    import re
    
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1).strip()
    else:
        json_str = raw
    
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback
        result = {
            "findings": [{
                "label": "Lab Report Analysis",
                "severity": "info",
                "confidence": 70,
                "region": "Full Report",
                "description": raw[:500],
            }],
            "impression": raw[:300],
            "pathology_scores": {},
            "detected_region": "lab_report",
        }
    
    # Ensure required fields
    result.setdefault("findings", [])
    result.setdefault("impression", "")
    result.setdefault("pathology_scores", {})
    result.setdefault("structures", [])
    result.setdefault("detected_region", "lab_report")
    
    return result


def _fallback_response(error_msg: str) -> dict:
    """Return safe fallback when all else fails."""
    return {
        "findings": [{
            "label": "Analysis Incomplete",
            "severity": "warning",
            "confidence": 50,
            "region": "System",
            "description": f"Could not complete automated analysis. Error: {error_msg[:200]}",
        }],
        "impression": "Automated analysis encountered an error. Please review the report manually.",
        "pathology_scores": {},
        "structures": [],
        "detected_region": "lab_report",
    }


def _merge_outputs(
    structured_result: Optional[dict],
    interpretation: dict,
    raw_text: str,
) -> dict:
    """Merge parser output with LLM interpretation."""
    output = {
        "modality": "lab_report",
        "findings": interpretation.get("findings", []),
        "impression": interpretation.get("impression", ""),
        "pathology_scores": interpretation.get("pathology_scores", {}),
        "structures": interpretation.get("structures", []),
        "detected_region": "lab_report",
    }
    
    # Add structured data if available
    if structured_result:
        output["structured"] = structured_result.get("structured", {})
        output["labs"] = structured_result.get("labs") or structured_result.get("structured", {})
        output["flattened_labs"] = structured_result.get("flattened_labs", {})
        output["document_type"] = structured_result.get("document_type", "lab_report")
        output["parser_confidence"] = structured_result.get("confidence", 0)
        output["parser_used"] = MEDGEMMA_PARSER_MODEL_ID
        output["pages_processed"] = structured_result.get("pages_processed", 1)
    else:
        output["structured"] = {"raw_text": raw_text[:5000]}
        output["labs"] = {}
        output["flattened_labs"] = {}
        output["parser_used"] = "text_only"
    
    # Add models used (LLM via OpenRouter)
    llm_label = "Manthana Report AI"
    if isinstance(interpretation, dict) and interpretation.get("_llm_model_used"):
        m = interpretation["_llm_model_used"]
        if "openrouter" in m.lower():
            llm_label = m.split(":", 1)[-1] if ":" in m else "OpenRouter"
        else:
            llm_label = m
    models_used = [output["parser_used"], llm_label]
    output["models_used"] = models_used
    if isinstance(interpretation, dict) and "_llm_model_used" in interpretation:
        output["llm_backend"] = interpretation.pop("_llm_model_used", None)
    
    # Add disclaimer
    output["disclaimer"] = (
        "This analysis is generated by AI and should be reviewed by a qualified "
        "healthcare professional. Always correlate with clinical findings."
    )
    
    return output


# ═════════════════════════════════════════════════════════════════════════════
# Legacy API Compatibility
# ═════════════════════════════════════════════════════════════════════════════

def analyze_lab_report_legacy(text: str) -> dict:
    """
    Legacy entry point for text-only analysis (backward compatibility).
    New code should use analyze_lab_report(filepath) for full capability.
    """
    interpretation = _interpret_raw_text(text, None)
    return _finalize_lab_output(_merge_outputs(None, interpretation, text))


def run_lab_report_pipeline_b64(
    document_b64: str,
    filename_hint: str = "report.pdf",
    clinical_notes: str = "",
    patient_context_json: str = "",
) -> dict:
    """Decode base64 → analyze + E2E enrichment (see inference.run_lab_report_pipeline_b64)."""
    from inference import run_lab_report_pipeline_b64 as _run_b64

    return _run_b64(
        document_b64,
        patient_context_json=patient_context_json,
        clinical_notes=clinical_notes,
        filename_hint=filename_hint,
    )


def is_service_ready() -> dict:
    """OPENROUTER_API_KEY required for readiness; MedGemma parser optional."""
    or_ok = False
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"):
        k = (os.environ.get(name) or "").strip()
        if k and len(k) >= 8:
            or_ok = True
            break
    return {
        "openrouter_configured": or_ok,
        "medgemma_parser_available": MEDGEMMA_PARSER_AVAILABLE,
        # Deprecated alias (same as medgemma_parser_available)
        "parrotlet_available": MEDGEMMA_PARSER_AVAILABLE,
        "ready": or_ok,
        "full_pipeline": or_ok and MEDGEMMA_PARSER_AVAILABLE,
    }
