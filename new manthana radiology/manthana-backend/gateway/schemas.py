"""
Manthana — Gateway Schemas
Request/response models specific to the gateway.
"""

from typing import Optional, List, Union
from pydantic import BaseModel


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


class CopilotRequest(BaseModel):
    question: str
    context: dict  # AnalysisResponse shape — treated as opaque dict
    language: str = "en"


class CopilotResponse(BaseModel):
    response: str
    model_used: str


