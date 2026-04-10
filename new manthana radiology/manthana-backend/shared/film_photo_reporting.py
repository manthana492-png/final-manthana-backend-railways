"""
Shared helpers for CT/MRI pipelines when input is mobile photos of printed film (film_photo_mode).
"""

from __future__ import annotations

from typing import Any, List

# Re-export multi-image extraction for use in CT/MRI pipelines
from preprocessing.film_photo_loader import extract_representative_slices_for_llm as extract_film_photo_images_for_llm

FILM_PHOTO_LIMITATION_NOTE = (
    "Analysis based on mobile phone photographs of printed CT/MRI film sheets — "
    "not original DICOM. Volumetric spacing, HU calibration, and segmentation are approximate only."
)

FILM_PHOTO_NARRATIVE_PREFIX = (
    "[FILM PHOTO INPUT — MULTI-IMAGE ANALYSIS MODE]\n\n"
    "You are analyzing 4-15 mobile phone photographs of printed CT or MRI film sheets. "
    "These represent sequential slices through the patient's brain. The local GPU pipeline "
    "could not perform reliable volumetric segmentation due to limited slice count and "
    "photographic artifacts (glare, perspective distortion, windowing variations).\n\n"
    "YOUR ROLE: Direct visual interpretation of the provided film photographs, combined with "
    "any available automated scores from the preprocessing pipeline.\n\n"
    "ANALYSIS GUIDANCE FOR FILM PHOTOS:\n"
    "1. ANATOMICAL ORIENTATION: Inferior-to-superior progression. First images are typically "
    "   more basal (brainstem, cerebellum), progressing toward vertex (cortical surfaces).\n"
    "2. WINDOWING AWARENESS: Film photos use fixed window settings from the original print. "
    "   Look for relative density differences rather than absolute HU values.\n"
    "3. PATHOLOGY PATTERNS TO ASSESS:\n"
    "   - Acute hemorrhage: Hyperdense (bright white) on CT films, variable on MRI depending on age\n"
    "   - Mass effect: Midline shift, sulcal effacement, ventricular compression\n"
    "   - Infarction: Hypodense (dark) wedge-shaped regions following vascular territories\n"
    "   - Mass lesions: Space-occupying lesions with mass effect, contrast enhancement if visible\n"
    "   - Hydrocephalus: Enlarged ventricles disproportionate to sulci\n"
    "   - Edema: Hypodense surrounding regions, loss of gray-white differentiation\n"
    "4. LIMITATIONS TO EMPHASIZE:\n"
    "   - Cannot measure precise volumes, densities, or subtle signal changes\n"
    "   - Limited sensitivity for small lesions (<1cm), posterior fossa, or subtle white matter changes\n"
    "   - No access to multiplanar reconstructions or contrast timing information\n"
    "   - Photographic glare may obscure pathology in some slices\n\n"
    "5. CLINICAL CORRELATION PRIORITIES:\n"
    "   - India-specific: Consider neurocysticercosis (NCC), tuberculoma, pyogenic abscess "
    "   - Emergency flags: Large hemorrhage, significant midline shift (>5mm), herniation signs\n"
    "   - Chronic changes: Atrophy patterns, old infarcts, calcifications\n\n"
    "OUTPUT REQUIREMENTS:\n"
    "- Lead with emergency findings if present (ICH, mass effect, herniation)\n"
    "- Describe anatomical level of any pathology (e.g., 'left MCA territory at basal ganglia level')\n"
    "- Qualitative assessments only: 'large', 'moderate', 'small', 'subtle' — never numeric measurements\n"
    "- Explicit uncertainty: 'limited by film photo quality' when findings are equivocal\n"
    "- Strong recommendation to obtain original DICOM for definitive diagnosis\n\n"
)


def is_film_photo_meta(meta: dict | None) -> bool:
    return bool(meta and meta.get("film_photo_mode"))


def apply_film_photo_pathology_scores(pathology_scores: dict[str, Any]) -> None:
    pathology_scores["film_photo_approximate"] = 1.0


def cap_confidence_for_film(confidence: str) -> str:
    """Never return high confidence for film-photo inputs."""
    c = (confidence or "medium").strip().lower()
    if c in ("high", "medium-high"):
        return "medium"
    return c


def merge_disclaimer_with_film(base: str, film: bool, film_addendum: str) -> str:
    if not film:
        return base
    return f"{base} {film_addendum}".strip()


def attach_film_meta_to_structures(structures: dict[str, Any], meta: dict | None) -> None:
    if not meta or not meta.get("film_photo_mode"):
        return
    structures["film_photo_mode"] = True
    structures["film_photo_quality"] = meta.get("film_photo_quality") or {}
    structures["limitation_note"] = (
        (structures.get("limitation_note") or "").strip() + " " + FILM_PHOTO_LIMITATION_NOTE
    ).strip()
