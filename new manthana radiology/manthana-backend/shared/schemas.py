"""
Manthana — Shared Pydantic Schemas
Unified request/response models used by all services.
"""

from typing import Optional, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
from enum import Enum


class Modality(str, Enum):
    XRAY = "xray"
    CHEST_XRAY = "chest_xray"
    BRAIN_MRI = "brain_mri"
    CARDIAC_CT = "cardiac_ct"
    PATHOLOGY = "pathology"
    ABDOMINAL_CT = "abdominal_ct"
    ULTRASOUND = "ultrasound"
    SPINE_NEURO = "spine_neuro"
    CYTOLOGY = "cytology"
    MAMMOGRAPHY = "mammography"
    ECG = "ecg"
    ORAL_CANCER = "oral_cancer"
    LAB_REPORT = "lab_report"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


# ─── Requests ───────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Request from gateway to individual service."""
    job_id: str
    modality: Modality
    patient_id: Optional[str] = None
    file_path: str  # Path to uploaded file in shared volume
    metadata: dict = Field(default_factory=dict)


class ReportRequest(BaseModel):
    """Request to report assembly service."""
    job_id: str
    modality: Modality
    findings: dict  # Raw model output
    structures: list = Field(default_factory=list)
    patient_id: Optional[str] = None


# ─── Responses ──────────────────────────────────────────

class Finding(BaseModel):
    """A single clinical finding — matches the frontend Finding interface."""
    label: str
    severity: str = "info"              # "critical" | "warning" | "info" | "clear"
    confidence: float = 0.0             # 0-100
    region: Optional[str] = None
    description: Optional[str] = None
    heatmap_url: Optional[str] = None   # Per-finding attention map

    @field_validator("severity")
    @classmethod
    def severity_must_be_contract(cls, v: str) -> str:
        allowed = {"critical", "warning", "info", "clear"}
        if v not in allowed:
            raise ValueError(f"severity must be one of {sorted(allowed)}, got {v!r}")
        return v


class AnalysisResponse(BaseModel):
    """Unified response from all analysis services."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    modality: str
    status: str = "complete"
    
    # Clinical output
    findings: List[Finding] = Field(default_factory=list)  # Structured findings array
    impression: str = ""                        # One-line summary
    pathology_scores: dict = Field(default_factory=dict)
    structures: Union[list, dict] = Field(
        default_factory=list,
        union_mode="left_to_right",
    )
    detected_region: Optional[str] = None       # For auto-detect (X-ray)
    
    # Confidence
    confidence: str = "medium"                  # high/medium/low
    confidence_score: Optional[float] = None    # 0.0-1.0
    
    # Visualization
    heatmap_url: Optional[str] = None
    segmentation_url: Optional[str] = None
    
    # Metadata
    processing_time_sec: float = 0.0
    models_used: list = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    is_critical: Optional[bool] = None

    # Lab report (optional — other modalities omit)
    labs: dict = Field(default_factory=dict)
    structured: dict = Field(default_factory=dict)
    parser_used: str = ""
    critical_values: List[str] = Field(default_factory=list)
    
    # Always included
    disclaimer: str = (
        "⚕️ DISCLAIMER: Manthana is an AI-assisted decision support tool "
        "for educational and second-opinion purposes only. It is NOT a "
        "certified medical device. All findings must be confirmed by a "
        "qualified radiologist or physician before clinical action. "
        "Manthana does not replace professional medical judgment."
    )


class JobResponse(BaseModel):
    """Response when a job is submitted to the queue."""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    position: Optional[int] = None
    estimated_seconds: Optional[int] = None
    message: str = "Job submitted successfully"


class JobStatusResponse(BaseModel):
    """Response when polling job status."""
    job_id: str
    status: JobStatus
    progress: Optional[float] = None  # 0.0-1.0
    result: Optional[AnalysisResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response for all services."""
    service: str
    status: str = "ok"
    models_loaded: bool = False
    gpu_available: bool = False
    version: str = "1.0.0"


# ─── Gateway & Unified Report Schemas ──────────────────────


class GatewayResponse(BaseModel):
    """Response from the gateway after submitting analysis."""

    model_config = ConfigDict(extra="allow")

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


class IndividualResult(BaseModel):
    """Single modality analysis result for unified report."""

    modality: str
    result: dict  # Full AnalysisResponse from the individual service


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
