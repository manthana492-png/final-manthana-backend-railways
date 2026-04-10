"""
Contradiction detection between quantitative scores and LLM narratives.
Cross-validates automated pipeline outputs against LLM-generated text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("manthana.contradiction_detector")


@dataclass
class Contradiction:
    """Represents a detected contradiction."""
    field: str
    automated_value: Any
    narrative_extracted: Any
    severity: str  # critical, warning, info
    message: str


class ContradictionDetector:
    """Detects contradictions between quantitative scores and LLM narratives."""
    
    # Thresholds for flagging contradictions
    CRITICAL_THRESHOLD = 0.3  # 30% difference triggers critical
    WARNING_THRESHOLD = 0.15  # 15% difference triggers warning
    
    def __init__(self):
        self.rules = self._build_rules()
    
    def _build_rules(self) -> Dict[str, Any]:
        """Build extraction and validation rules per modality."""
        return {
            "brain_mri": {
                "extractors": {
                    "midline_shift": self._extract_midline_shift,
                    "hydrocephalus": self._extract_hydrocephalus,
                    "mass_effect": self._extract_mass_effect,
                    "lesion_present": self._extract_lesion_presence,
                },
                "comparators": {
                    "midline_shift": self._compare_midline_shift,
                    "hydrocephalus": self._compare_boolean,
                    "mass_effect": self._compare_boolean,
                    "lesion_present": self._compare_boolean,
                },
            },
            "ct_brain": {
                "extractors": {
                    "hemorrhage_present": self._extract_hemorrhage,
                    "midline_shift": self._extract_midline_shift,
                    "hydrocephalus": self._extract_hydrocephalus,
                    "hemorrhage_volume": self._extract_hemorrhage_volume,
                },
                "comparators": {
                    "hemorrhage_present": self._compare_boolean,
                    "midline_shift": self._compare_midline_shift,
                    "hydrocephalus": self._compare_boolean,
                    "hemorrhage_volume": self._compare_volume,
                },
            },
            "chest_ct": {
                "extractors": {
                    "nodules_present": self._extract_nodules,
                    "consolidation": self._extract_consolidation,
                    "effusion": self._extract_effusion,
                    "cavitation": self._extract_cavitation,
                },
                "comparators": {
                    "nodules_present": self._compare_boolean,
                    "consolidation": self._compare_boolean,
                    "effusion": self._compare_boolean,
                    "cavitation": self._compare_boolean,
                },
            },
            "cardiac_ct": {
                "extractors": {
                    "calcium_score": self._extract_calcium_score,
                    "pericardial_effusion": self._extract_pericardial_effusion,
                    "cardiomegaly": self._extract_cardiomegaly,
                },
                "comparators": {
                    "calcium_score": self._compare_calcium_score,
                    "pericardial_effusion": self._compare_boolean,
                    "cardiomegaly": self._compare_boolean,
                },
            },
            "spine_ct": {
                "extractors": {
                    "fracture_present": self._extract_fracture,
                    "stenosis": self._extract_stenosis,
                    "spondylolisthesis": self._extract_spondy,
                },
                "comparators": {
                    "fracture_present": self._compare_boolean,
                    "stenosis": self._compare_boolean,
                    "spondylolisthesis": self._compare_boolean,
                },
            },
            "spine_mri": {
                "extractors": {
                    "fracture_present": self._extract_fracture,
                    "cord_compression": self._extract_cord_compression,
                    "pott_disease": self._extract_pott,
                    "modic_changes": self._extract_modic,
                },
                "comparators": {
                    "fracture_present": self._compare_boolean,
                    "cord_compression": self._compare_boolean,
                    "pott_disease": self._compare_boolean,
                    "modic_changes": self._compare_boolean,
                },
            },
        }
    
    # Extractor methods for common findings
    def _extract_midline_shift(self, narrative: str) -> Optional[float]:
        """Extract midline shift measurement from narrative."""
        # Look for patterns like "midline shift of 5mm", "5 mm midline shift"
        patterns = [
            r"midline\s+shift\s+of\s+(\d+\.?\d*)\s*mm",
            r"midline\s+shift\s+(\d+\.?\d*)\s*mm",
            r"(\d+\.?\d*)\s*mm\s+midline\s+shift",
            r"shift\s+of\s+(\d+\.?\d*)\s*mm",
        ]
        for pattern in patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def _extract_hydrocephalus(self, narrative: str) -> Optional[bool]:
        """Extract hydrocephalus presence from narrative."""
        positive = r"(hydrocephalus|ventricular\s+enlargement|enlarged\s+ventricles)"
        negative = r"(no\s+hydrocephalus|no\s+ventricular\s+enlargement|ventricles\s+normal)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_mass_effect(self, narrative: str) -> Optional[bool]:
        """Extract mass effect presence from narrative."""
        positive = r"(mass\s+effect|mass-effect|mass\s+with\s+effect)"
        negative = r"(no\s+mass\s+effect|no\s+significant\s+mass)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_lesion_presence(self, narrative: str) -> Optional[bool]:
        """Extract lesion presence from narrative."""
        positive = r"(lesion|tumor|mass|enhancing|abnormality\s+in\s+the\s+brain)"
        negative = r"(no\s+lesion|no\s+tumor|no\s+mass|normal\s+parenchyma)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_hemorrhage(self, narrative: str) -> Optional[bool]:
        """Extract hemorrhage presence from narrative."""
        positive = r"(hemorrhage|hematoma|bleed|hyperdense\s+signal|acute\s+blood)"
        negative = r"(no\s+hemorrhage|no\s+acute\s+bleed|no\s+hematoma)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_hemorrhage_volume(self, narrative: str) -> Optional[float]:
        """Extract hemorrhage volume from narrative."""
        patterns = [
            r"volume\s+of\s+(\d+\.?\d*)\s*ml",
            r"(\d+\.?\d*)\s*ml\s+(volume|hemorrhage|hematoma)",
            r"approximately\s+(\d+\.?\d*)\s*ml",
        ]
        for pattern in patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def _extract_nodules(self, narrative: str) -> Optional[bool]:
        """Extract nodule presence from narrative."""
        positive = r"(nodule|pulmonary\s+nodule|lung\s+nodule)"
        negative = r"(no\s+nodule|no\s+pulmonary\s+nodules)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_consolidation(self, narrative: str) -> Optional[bool]:
        """Extract consolidation presence from narrative."""
        positive = r"(consolidation|airspace\s+opacification|infiltrate)"
        negative = r"(no\s+consolidation|no\s+airspace\s+opacification)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_effusion(self, narrative: str) -> Optional[bool]:
        """Extract pleural effusion presence from narrative."""
        positive = r"(pleural\s+effusion|effusion)"
        negative = r"(no\s+pleural\s+effusion|no\s+effusion)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_cavitation(self, narrative: str) -> Optional[bool]:
        """Extract cavitation presence from narrative."""
        positive = r"(cavit|cavity|pneumatocele)"
        negative = r"(no\s+cavit|no\s+cavity)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_calcium_score(self, narrative: str) -> Optional[float]:
        """Extract calcium score from narrative."""
        patterns = [
            r"agatston\s+score\s+of\s+(\d+)",
            r"calcium\s+score\s+(\d+)",
            r"cac\s+score\s+(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def _extract_pericardial_effusion(self, narrative: str) -> Optional[bool]:
        """Extract pericardial effusion from narrative."""
        positive = r"(pericardial\s+effusion|pericardial\s+fluid)"
        negative = r"(no\s+pericardial\s+effusion|no\s+pericardial\s+fluid)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_cardiomegaly(self, narrative: str) -> Optional[bool]:
        """Extract cardiomegaly from narrative."""
        positive = r"(cardiomegaly|enlarged\s+cardiac|cardiomegal)"
        negative = r"(no\s+cardiomegaly|normal\s+cardiac\s+size)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_fracture(self, narrative: str) -> Optional[bool]:
        """Extract fracture presence from narrative."""
        positive = r"(fracture|compression\s+fracture|burst\s+fracture|vertebral\s+fracture)"
        negative = r"(no\s+fracture|no\s+acute\s+fracture|no\s+compression\s+fracture)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_stenosis(self, narrative: str) -> Optional[bool]:
        """Extract stenosis presence from narrative."""
        positive = r"(stenosis|spinal\s+stenosis|canal\s+stenosis)"
        negative = r"(no\s+stenosis|no\s+significant\s+stenosis)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_spondy(self, narrative: str) -> Optional[bool]:
        """Extract spondylolisthesis from narrative."""
        positive = r"(spondylolisthesis|listhesis|anterolisthesis|retrolisthesis)"
        negative = r"(no\s+spondylolisthesis|normal\s+alignment)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_cord_compression(self, narrative: str) -> Optional[bool]:
        """Extract cord compression from narrative."""
        positive = r"(cord\s+compression|spinal\s+cord\s+compression|myelopathy)"
        negative = r"(no\s+cord\s+compression|no\s+spinal\s+cord\s+compression)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_pott(self, narrative: str) -> Optional[bool]:
        """Extract Pott disease from narrative."""
        positive = r"(pott['’]?s\s*disease|tb\s*spine|tuberculosis|spinal\s*tb)"
        negative = r"(no\s+evidence\s+of\s*pott|no\s+tb\s+spine)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    def _extract_modic(self, narrative: str) -> Optional[bool]:
        """Extract Modic changes from narrative."""
        positive = r"(modic\s+type|modic\s+change|endplate\s+edema|endplate\s+signal)"
        negative = r"(no\s+modic\s+changes|no\s+endplate\s+edema)"
        
        if re.search(positive, narrative, re.IGNORECASE):
            return True
        if re.search(negative, narrative, re.IGNORECASE):
            return False
        return None
    
    # Comparator methods
    def _compare_midline_shift(
        self,
        auto_value: Optional[float],
        narrative_value: Optional[float],
    ) -> Tuple[bool, Optional[Contradiction]]:
        """Compare midline shift values."""
        if auto_value is None or narrative_value is None:
            return True, None
        
        diff_pct = abs(auto_value - narrative_value) / max(auto_value, 1.0)
        
        if diff_pct > self.CRITICAL_THRESHOLD:
            return False, Contradiction(
                field="midline_shift",
                automated_value=auto_value,
                narrative_extracted=narrative_value,
                severity="critical",
                message=f"Midline shift differs significantly: automated={auto_value:.1f}mm, narrative={narrative_value:.1f}mm",
            )
        elif diff_pct > self.WARNING_THRESHOLD:
            return False, Contradiction(
                field="midline_shift",
                automated_value=auto_value,
                narrative_extracted=narrative_value,
                severity="warning",
                message=f"Midline shift differs moderately: automated={auto_value:.1f}mm, narrative={narrative_value:.1f}mm",
            )
        return True, None
    
    def _compare_boolean(
        self,
        auto_value: Optional[bool],
        narrative_value: Optional[bool],
    ) -> Tuple[bool, Optional[Contradiction]]:
        """Compare boolean values."""
        if auto_value is None or narrative_value is None:
            return True, None
        
        if auto_value != narrative_value:
            severity = "critical" if auto_value else "warning"
            return False, Contradiction(
                field="boolean_field",
                automated_value=auto_value,
                narrative_extracted=narrative_value,
                severity=severity,
                message=f"Boolean mismatch: automated={auto_value}, narrative={narrative_value}",
            )
        return True, None
    
    def _compare_volume(
        self,
        auto_value: Optional[float],
        narrative_value: Optional[float],
    ) -> Tuple[bool, Optional[Contradiction]]:
        """Compare volume values."""
        if auto_value is None or narrative_value is None:
            return True, None
        
        diff_pct = abs(auto_value - narrative_value) / max(auto_value, 1.0)
        
        if diff_pct > self.CRITICAL_THRESHOLD:
            return False, Contradiction(
                field="volume",
                automated_value=auto_value,
                narrative_extracted=narrative_value,
                severity="critical",
                message=f"Volume differs significantly: automated={auto_value:.1f}ml, narrative={narrative_value:.1f}ml",
            )
        elif diff_pct > self.WARNING_THRESHOLD:
            return False, Contradiction(
                field="volume",
                automated_value=auto_value,
                narrative_extracted=narrative_value,
                severity="warning",
                message=f"Volume differs moderately: automated={auto_value:.1f}ml, narrative={narrative_value:.1f}ml",
            )
        return True, None
    
    def _compare_calcium_score(
        self,
        auto_value: Optional[float],
        narrative_value: Optional[float],
    ) -> Tuple[bool, Optional[Contradiction]]:
        """Compare calcium scores (more lenient due to estimation methods)."""
        if auto_value is None or narrative_value is None:
            return True, None
        
        # For calcium scores, use absolute difference
        diff = abs(auto_value - narrative_value)
        diff_pct = diff / max(auto_value, 100.0)  # Use 100 as minimum denominator
        
        if diff_pct > 0.5:  # 50% difference threshold for calcium
            return False, Contradiction(
                field="calcium_score",
                automated_value=auto_value,
                narrative_extracted=narrative_value,
                severity="warning",
                message=f"Calcium score differs: automated={auto_value:.0f}, narrative={narrative_value:.0f}",
            )
        return True, None
    
    def detect_contradictions(
        self,
        modality: str,
        automated_scores: Dict[str, Any],
        narrative: str,
    ) -> List[Contradiction]:
        """
        Detect contradictions between automated scores and narrative.
        
        Args:
            modality: Imaging modality (brain_mri, ct_brain, etc.)
            automated_scores: Dict of automated pipeline scores
            narrative: LLM-generated narrative text
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        rules = self.rules.get(modality, {})
        extractors = rules.get("extractors", {})
        comparators = rules.get("comparators", {})
        
        for field, extractor in extractors.items():
            try:
                narrative_value = extractor(narrative)
                auto_value = automated_scores.get(field)
                
                comparator = comparators.get(field, self._compare_boolean)
                is_consistent, contradiction = comparator(auto_value, narrative_value)
                
                if not is_consistent and contradiction:
                    contradictions.append(contradiction)
            except Exception as e:
                logger.debug(f"Failed to check {field}: {e}")
        
        return contradictions


def check_narrative_consistency(
    modality: str,
    automated_scores: Dict[str, Any],
    narrative: str,
) -> Dict[str, Any]:
    """
    Convenience function to check narrative consistency.
    
    Returns dict with:
        consistent: bool
        contradictions: list of contradiction dicts
        summary: str
    """
    detector = ContradictionDetector()
    contradictions = detector.detect_contradictions(modality, automated_scores, narrative)
    
    if not contradictions:
        return {
            "consistent": True,
            "contradictions": [],
            "summary": "No contradictions detected between automated scores and narrative",
        }
    
    critical_count = sum(1 for c in contradictions if c.severity == "critical")
    warning_count = sum(1 for c in contradictions if c.severity == "warning")
    
    if critical_count > 0:
        summary = f"CRITICAL: {critical_count} major contradiction(s) detected between automated measurements and narrative text"
    elif warning_count > 0:
        summary = f"WARNING: {warning_count} moderate inconsistency(s) detected - review recommended"
    else:
        summary = "Minor inconsistencies detected - likely acceptable variance"
    
    return {
        "consistent": critical_count == 0,
        "contradictions": [
            {
                "field": c.field,
                "automated_value": c.automated_value,
                "narrative_extracted": c.narrative_extracted,
                "severity": c.severity,
                "message": c.message,
            }
            for c in contradictions
        ],
        "summary": summary,
        "critical_count": critical_count,
        "warning_count": warning_count,
    }