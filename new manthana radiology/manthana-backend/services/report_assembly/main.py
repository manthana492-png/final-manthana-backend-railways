"""
Manthana — Report Assembly Service
Generates narrative radiology reports using LLM APIs.
DeepSeek V3 → Gemini Flash → Groq → Qwen 2.5
Supports 10 Indian languages natively.
"""

import json
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from assembler import generate_report, generate_unified_report
from correlation_engine import find_correlations
from prompts import SUPPORTED_LANGUAGES, get_language_config
from pdf_renderer import render_report_pdf, PDF_OUTPUT_DIR
from config import SERVICE_NAME, PORT

app = FastAPI(title=f"Manthana — {SERVICE_NAME}")


class ReportRequest(BaseModel):
    job_id: str
    modality: str
    findings: dict
    structures: list = []
    patient_id: Optional[str] = None
    detected_region: Optional[str] = None
    language: str = "en"  # ISO language code: en, hi, ta, te, kn, ml, mr, bn, gu, pa
    generate_pdf: bool = False  # Whether to also render a PDF


class ReportResponse(BaseModel):
    job_id: str
    narrative: str
    impression: str
    models_used: list
    processing_time_sec: float
    language: str = "en"
    language_fallback: bool = False  # True if LLM fell back to English
    pdf_url: Optional[str] = None


# ── Unified Report Models ──

class IndividualReportItem(BaseModel):
    modality: str
    result: dict


class UnifiedReportRequest(BaseModel):
    results: List[IndividualReportItem]
    patient_id: Optional[str] = None
    language: str = "en"  # ISO language code


class CorrelationItem(BaseModel):
    pattern: str
    confidence: float
    clinical_significance: str
    matching_modalities: List[str]
    action: str


class UnifiedReportResponse(BaseModel):
    patient_id: str
    modalities_analyzed: List[str]
    individual_reports: List[dict]
    unified_diagnosis: str
    unified_findings: str
    risk_assessment: str
    treatment_recommendations: str
    prognosis: str
    cross_modality_correlations: str
    confidence: str
    models_used: List[str]
    processing_time_sec: float
    correlations: Optional[List[CorrelationItem]] = None


@app.get("/health")
async def health():
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "version": "1.0.0",
    }


@app.get("/languages")
async def list_languages():
    """List all supported report languages."""
    return {
        "supported": [
            {"code": code, **{k: v for k, v in get_language_config(code).items()
                             if k in ("name", "script", "direction")}}
            for code in SUPPORTED_LANGUAGES
        ]
    }


@app.post("/assemble_report", response_model=ReportResponse)
async def assemble_report(request: ReportRequest):
    """Generate a narrative radiology report from raw findings."""
    start = time.time()

    try:
        narrative, impression, llm_used = await generate_report(
            modality=request.modality,
            findings=request.findings,
            structures=request.structures,
            detected_region=request.detected_region,
            language=request.language,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    language_fallback = llm_used == "fallback-en" and request.language != "en"

    pdf_url: Optional[str] = None
    if request.generate_pdf:
        # Build language-specific headers and disclaimer for PDF rendering
        lang_cfg = get_language_config(request.language)
        findings_header = lang_cfg.get("findings_header", "FINDINGS")
        impression_header = lang_cfg.get("impression_header", "IMPRESSION")
        disclaimer = lang_cfg.get(
            "disclaimer",
            "This is an AI-generated second opinion, not a primary diagnosis.",
        )
        try:
            pdf_path = await render_report_pdf(
                narrative=narrative,
                impression=impression,
                modality=request.modality,
                patient_id=request.patient_id or "ANONYMOUS",
                language=request.language,
                findings_header=findings_header,
                impression_header=impression_header,
                disclaimer=disclaimer,
            )
            if pdf_path:
                filename = os.path.basename(pdf_path)
                pdf_url = f"/reports/{filename}"
        except Exception as e:
            # PDF failure is non-fatal — log and continue
            import logging

            logging.getLogger("manthana.report_assembly").warning(
                "PDF render failed for job %s: %s", request.job_id, e
            )

    return ReportResponse(
        job_id=request.job_id,
        narrative=narrative,
        impression=impression,
        models_used=[llm_used],
        processing_time_sec=round(time.time() - start, 2),
        language=request.language,
        language_fallback=language_fallback,
        pdf_url=pdf_url,
    )


@app.post("/assemble_unified_report", response_model=UnifiedReportResponse)
async def assemble_unified_report(request: UnifiedReportRequest):
    """Generate a unified cross-modality report from individual modality analyses."""
    start = time.time()

    # Extract individual report summaries for the LLM
    individual_reports = []
    raw_for_corr = []
    for item in request.results:
        result = item.result
        findings_summary = ""
        if isinstance(result.get("findings"), list):
            findings_summary = "; ".join(
                f.get("label", "") for f in result["findings"] if isinstance(f, dict)
            )
        elif isinstance(result.get("findings"), str):
            findings_summary = result["findings"]

        individual_reports.append({
            "modality": item.modality,
            "findings_summary": findings_summary or "No findings available",
            "impression": result.get("impression", "No impression available"),
        })
        raw_for_corr.append({"modality": item.modality, "result": result})

    correlations = find_correlations(raw_for_corr)
    corr_block = json.dumps(correlations, indent=2) if correlations else ""

    try:
        unified = await generate_unified_report(
            individual_reports,
            language=request.language,
            correlations_block=corr_block,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unified report generation failed: {str(e)}",
        )

    return UnifiedReportResponse(
        patient_id=request.patient_id or "ANONYMOUS",
        modalities_analyzed=[item.modality for item in request.results],
        individual_reports=individual_reports,
        unified_diagnosis=unified.get("unified_diagnosis", ""),
        unified_findings=unified.get("unified_findings", ""),
        risk_assessment=unified.get("risk_assessment", ""),
        treatment_recommendations=unified.get("treatment_recommendations", ""),
        prognosis=unified.get("prognosis", ""),
        cross_modality_correlations=unified.get("cross_modality_correlations", ""),
        confidence=unified.get("confidence", "moderate"),
        models_used=unified.get("models_used", []),
        processing_time_sec=round(time.time() - start, 2),
        correlations=[CorrelationItem(**c) for c in correlations] if correlations else None,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

