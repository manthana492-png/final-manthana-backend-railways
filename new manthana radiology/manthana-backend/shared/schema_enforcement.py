"""
Pydantic schema enforcement for LLM outputs per modality.
Uses instructor library for structured LLM output validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    CLEAR = "clear"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FindingSchema(BaseModel):
    """Schema for a single imaging finding."""
    label: str = Field(..., description="Finding label/name")
    description: str = Field(..., description="Detailed description")
    severity: SeverityLevel = Field(..., description="Severity level")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence 0-100")
    region: str = Field(..., description="Anatomical region")


class ImpressionSchema(BaseModel):
    """Schema for study impression."""
    summary: str = Field(..., description="Brief summary of key findings")
    recommendation: str = Field(..., description="Clinical recommendations")
    urgency: Literal["routine", "urgent", "stat"] = Field(..., description="Urgency level")


# Brain MRI Schema
class BrainLesionSchema(BaseModel):
    """Schema for brain lesion description."""
    present: bool = Field(..., description="Whether lesion is present")
    location: Optional[str] = Field(None, description="Lesion anatomical location")
    size_mm: Optional[float] = Field(None, ge=0, description="Lesion size in mm")
    type: Optional[str] = Field(None, description="Lesion type (tumor, stroke, etc)")
    enhancement_pattern: Optional[str] = Field(None, description="Enhancement pattern if contrast given")


class BrainMRIReportSchema(BaseModel):
    """Structured brain MRI report schema."""
    modality: Literal["brain_mri"] = "brain_mri"
    technique: str = Field(..., description="Imaging technique summary")
    comparison: Optional[str] = Field(None, description="Comparison to prior studies")
    
    findings: List[FindingSchema] = Field(default_factory=list)
    
    # Specific brain findings
    ventricles: Optional[str] = Field(None, description="Ventricle size and morphology")
    midline_shift: Optional[bool] = Field(None, description="Midline shift present")
    white_matter: Optional[str] = Field(None, description="White matter signal assessment")
    gray_matter: Optional[str] = Field(None, description="Gray matter assessment")
    mass_effect: Optional[bool] = Field(None, description="Mass effect present")
    herniation: Optional[bool] = Field(None, description="Herniation signs")
    
    lesions: List[BrainLesionSchema] = Field(default_factory=list)
    
    impression: ImpressionSchema
    
    model_config = {"json_schema_extra": {"title": "Brain MRI Structured Report"}}


# CT Brain Schema
class HemorrhageSchema(BaseModel):
    """Schema for hemorrhage findings."""
    present: bool = Field(..., description="Hemorrhage detected")
    type: Optional[str] = Field(None, description="Hemorrhage type (ICH, SDH, EDH, SAH, IVH)")
    location: Optional[str] = Field(None, description="Anatomical location")
    volume_ml: Optional[float] = Field(None, ge=0, description="Estimated volume in ml")
    mass_effect: Optional[bool] = Field(None, description="Causes mass effect")


class CTBrainReportSchema(BaseModel):
    """Structured CT brain report schema."""
    modality: Literal["ct_brain"] = "ct_brain"
    technique: str = Field(..., description="Non-contrast CT brain")
    comparison: Optional[str] = Field(None, description="Comparison to prior")
    
    findings: List[FindingSchema] = Field(default_factory=list)
    
    # Specific CT findings
    hemorrhage: Optional[HemorrhageSchema] = Field(None, description="Hemorrhage assessment")
    midline_shift_mm: Optional[float] = Field(None, ge=0, description="Midline shift in mm")
    ventricle_size: Optional[str] = Field(None, description="Ventricle size assessment")
    hydrocephalus: Optional[bool] = Field(None, description="Hydrocephalus present")
    sulci: Optional[str] = Field(None, description="Sulcal effacement/prominence")
    gray_white_differentiation: Optional[str] = Field(None, description="Gray-white differentiation")
    
    impression: ImpressionSchema
    
    model_config = {"json_schema_extra": {"title": "CT Brain Structured Report"}}


# Chest CT Schema
class LungNoduleSchema(BaseModel):
    """Schema for lung nodule."""
    present: bool = Field(..., description="Nodule present")
    location: Optional[str] = Field(None, description="Lobe/segment location")
    size_mm: Optional[float] = Field(None, ge=0, description="Size in mm")
    density: Optional[str] = Field(None, description="Solid, part-solid, or ground-glass")
    risk_category: Optional[str] = Field(None, description="Risk category if calculated")


class ChestCTReportSchema(BaseModel):
    """Structured chest CT report schema."""
    modality: Literal["chest_ct"] = "chest_ct"
    technique: str = Field(..., description="CT chest technique")
    
    findings: List[FindingSchema] = Field(default_factory=list)
    
    # Lung findings
    lung_parenchyma: Optional[str] = Field(None, description="Parenchymal assessment")
    nodules: List[LungNoduleSchema] = Field(default_factory=list)
    consolidation: Optional[str] = Field(None, description="Consolidation assessment")
    cavitation: Optional[str] = Field(None, description="Cavitation if present")
    
    # Pleural findings
    pleura: Optional[str] = Field(None, description="Pleural assessment")
    effusion: Optional[str] = Field(None, description="Pleural effusion if present")
    
    # Mediastinal findings
    mediastinum: Optional[str] = Field(None, description="Mediastinal assessment")
    lymph_nodes: Optional[str] = Field(None, description="Lymph node assessment")
    
    # Cardiac findings (visible on chest CT)
    heart_size: Optional[str] = Field(None, description="Heart size assessment")
    pericardium: Optional[str] = Field(None, description="Pericardial assessment")
    
    # Abdomen (upper)
    upper_abdomen: Optional[str] = Field(None, description="Upper abdomen assessment if visible")
    
    impression: ImpressionSchema
    
    model_config = {"json_schema_extra": {"title": "Chest CT Structured Report"}}


# Cardiac CT Schema
class CalciumScoreSchema(BaseModel):
    """Schema for coronary calcium."""
    agatston_score: Optional[float] = Field(None, ge=0, description="Agatston calcium score")
    risk_category: Optional[str] = Field(None, description="Risk stratification")
    vessels_affected: List[str] = Field(default_factory=list, description="Affected vessels")


class CardiacCTReportSchema(BaseModel):
    """Structured cardiac CT report schema."""
    modality: Literal["cardiac_ct"] = "cardiac_ct"
    technique: str = Field(..., description="CT cardiac technique")
    
    findings: List[FindingSchema] = Field(default_factory=list)
    
    # Cardiac structures
    heart_size: Optional[str] = Field(None, description="Heart size assessment")
    chambers: Optional[str] = Field(None, description="Chamber assessment")
    pericardium: Optional[str] = Field(None, description="Pericardial assessment")
    
    # Aorta
    aorta_size: Optional[str] = Field(None, description="Aortic size assessment")
    aortic_calcification: Optional[str] = Field(None, description="Aortic calcification")
    
    # Coronary
    coronary_calcium: Optional[CalciumScoreSchema] = Field(None, description="Coronary calcium")
    stenosis: Optional[str] = Field(None, description="Stenosis assessment if contrast")
    
    impression: ImpressionSchema
    
    model_config = {"json_schema_extra": {"title": "Cardiac CT Structured Report"}}


# Spine CT/MRI Schema
class VertebralLevelSchema(BaseModel):
    """Schema for vertebral level assessment."""
    level: str = Field(..., description="Vertebral level (e.g., L1, T12)")
    alignment: str = Field(default="normal", description="Alignment status")
    height_loss_pct: Optional[float] = Field(None, ge=0, le=100, description="Height loss %")
    fracture_grade: Optional[int] = Field(None, ge=0, le=3, description="Genant fracture grade")
    marrow_signal: Optional[str] = Field(None, description="Marrow signal (MRI)")
    endplate_changes: Optional[str] = Field(None, description="Endplate degenerative changes")


class SpineReportSchema(BaseModel):
    """Structured spine CT/MRI report schema."""
    modality: Literal["spine_ct", "spine_mri"] = "spine_mri"
    technique: str = Field(..., description="Spine imaging technique")
    
    findings: List[FindingSchema] = Field(default_factory=list)
    
    # Alignment
    alignment: Optional[str] = Field(None, description="Spinal alignment")
    scoliosis: Optional[str] = Field(None, description="Scoliosis assessment")
    spondylolisthesis: Optional[str] = Field(None, description="Spondylolisthesis assessment")
    
    # Vertebral bodies
    vertebral_levels: List[VertebralLevelSchema] = Field(default_factory=list)
    
    # Discs
    disc_spaces: Optional[str] = Field(None, description="Disc space assessment")
    disc_degeneration: Optional[str] = Field(None, description="Degenerative changes")
    
    # Spinal canal
    canal_stenosis: Optional[str] = Field(None, description="Canal stenosis assessment")
    cord_compression: Optional[str] = Field(None, description="Cord compression (MRI)")
    
    # Soft tissues
    paraspinal_soft_tissues: Optional[str] = Field(None, description="Paraspinal assessment")
    
    impression: ImpressionSchema
    
    model_config = {"json_schema_extra": {"title": "Spine Imaging Structured Report"}}


# Modality to schema mapping
MODALITY_SCHEMAS = {
    "brain_mri": BrainMRIReportSchema,
    "ct_brain": CTBrainReportSchema,
    "chest_ct": ChestCTReportSchema,
    "cardiac_ct": CardiacCTReportSchema,
    "spine_ct": SpineReportSchema,
    "spine_mri": SpineReportSchema,
}


def get_schema_for_modality(modality: str) -> type[BaseModel]:
    """Get the Pydantic schema class for a given modality."""
    return MODALITY_SCHEMAS.get(modality, BaseModel)


def validate_llm_output(output: dict, modality: str) -> tuple[bool, Union[BaseModel, str]]:
    """
    Validate LLM output against modality-specific schema.
    
    Returns:
        (is_valid, validated_model_or_error_message)
    """
    schema_class = get_schema_for_modality(modality)
    if schema_class is BaseModel:
        return True, output  # No specific schema, pass through
    
    try:
        validated = schema_class(**output)
        return True, validated
    except Exception as e:
        return False, str(e)


class StructuredReport(BaseModel):
    """Generic structured report container with modality-specific content."""
    modality: str = Field(..., description="Imaging modality")
    schema_version: str = Field(default="1.0")
    content: Union[
        BrainMRIReportSchema,
        CTBrainReportSchema,
        ChestCTReportSchema,
        CardiacCTReportSchema,
        SpineReportSchema,
        Dict[str, Any],  # Fallback for unknown modalities
    ] = Field(..., description="Modality-specific structured report content")