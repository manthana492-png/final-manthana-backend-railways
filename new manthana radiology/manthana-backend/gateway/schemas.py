"""
Manthana — Gateway Schemas
Request/response models specific to the gateway.
"""

from typing import Any, Optional, List, Union
from pydantic import BaseModel, ConfigDict, Field


class Finding(BaseModel):
    """A single clinical finding — matches the frontend Finding interface."""
    label: str
    severity: str = "info"              # "critical" | "warning" | "info" | "clear"
    confidence: float = 0.0             # 0-100
    region: Optional[str] = None
    description: Optional[str] = None
    heatmap_url: Optional[str] = None   # Per-finding attention map


class GatewayResponse(BaseModel):
    """Response from the gateway after submitting analysis."""

    model_config = {"extra": "allow"}

    job_id: str
    status: str = "complete"
    modality: Optional[str] = None
    findings: Optional[List[Finding]] = None
    impression: Optional[str] = None
    pathology_scores: Optional[dict] = None
    structures: Optional[Union[list, dict]] = None
    detected_region: Optional[str] = None
    confidence: Optional[str] = None
    confidence_score: Optional[float] = None
    heatmap_url: Optional[str] = None
    processing_time_sec: Optional[float] = None
    models_used: Optional[list] = None
    disclaimer: Optional[str] = None
    message: Optional[str] = None
    analysis_depth: Optional[str] = None
    ensemble_agreement: Optional[float] = None


# ── Multi-Model Unified Report Schemas ──

class IndividualResult(BaseModel):
    """Single modality analysis result for unified report."""
    modality: str
    result: dict   # Full AnalysisResponse from the individual service


class UnifiedReportRequest(BaseModel):
    """Request for unified cross-modality report."""
    results: List[IndividualResult]
    patient_id: Optional[str] = None
    language: str = "en"


class SingleReportRequest(BaseModel):
    """Single-modality narrative report — maps frontend AnalysisResponse shape to report_assembly."""

    model_config = ConfigDict(extra="allow")

    modality: str
    findings: Union[List[Any], dict] = Field(default_factory=list)
    impression: str = ""
    pathology_scores: dict = Field(default_factory=dict)
    structures: Union[list, dict] = Field(default_factory=list)
    patient_id: Optional[str] = None
    clinical_notes: Optional[str] = None
    language: str = "en"
    detected_region: Optional[str] = None
    job_id: Optional[str] = None


class CopilotRequest(BaseModel):
    question: str
    context: dict  # AnalysisResponse shape — treated as opaque dict
    language: str = "en"


class CopilotResponse(BaseModel):
    response: str
    model_used: str


class CorrelationItem(BaseModel):
    pattern: str
    confidence: float
    clinical_significance: str
    matching_modalities: List[str]
    action: str


class UnifiedReportResponse(BaseModel):
    """Response with unified cross-modality analysis."""
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

