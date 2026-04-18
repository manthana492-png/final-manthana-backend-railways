"""
95-modality registry — SSOT for AI orchestration (interrogator / interpreter YAML roles).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

# Group id -> interrogator role name in cloud_inference.yaml (orch_chains key).
# All groups use one universal chain: Kimi K2.5 Thinking + NIM vision + Qwen + terminal Kimi.
INTERROGATOR_ROLE_BY_GROUP: Dict[str, str] = {
    "xray": "interrogator_universal",
    "ct": "interrogator_universal",
    "mri": "interrogator_universal",
    "nuclear": "interrogator_universal",
    "ultrasound": "interrogator_universal",
    "cardiac_functional": "interrogator_universal",
    "specialized": "interrogator_universal",
    "pathology": "interrogator_universal",
    "oncology": "interrogator_universal",
    "ophthalmology_dental": "interrogator_universal",
    "reports": "interrogator_universal",
}

MODALITY_GROUPS_META: Dict[str, Dict[str, object]] = {
    "xray": {"label": "X-Ray", "icon": "XRAY", "order": 1},
    "ct": {"label": "CT Scans", "icon": "CT", "order": 2},
    "mri": {"label": "MRI", "icon": "MRI", "order": 3},
    "nuclear": {"label": "Nuclear / PET", "icon": "NUC", "order": 4},
    "ultrasound": {"label": "Ultrasound", "icon": "USG", "order": 5},
    "cardiac_functional": {"label": "Cardiac / Functional", "icon": "ECG", "order": 6},
    "specialized": {"label": "Specialized Imaging", "icon": "SPEC", "order": 7},
    "pathology": {"label": "Pathology / Cytology", "icon": "PATH", "order": 8},
    "oncology": {"label": "Oncology Imaging", "icon": "ONC", "order": 9},
    "ophthalmology_dental": {"label": "Ophthalmology / Dental", "icon": "EYE", "order": 10},
    "reports": {"label": "Reports & Documents", "icon": "DOC", "order": 11},
}

# Modalities that use cardiac interpreter (Kimi thinking etc.)
_INTERPRETER_CARDIAC_KEYS: FrozenSet[str] = frozenset(
    {
        "echo_cardiac",
        "cardiac_mri",
        "ct_cardiac",
        "ct_angiography",
        "ct_perfusion",
        "mri_perfusion",
        "myocardial_perf",
        "ecg",
        "holter_monitor",
    }
)

# Pathology / oncology interpreter
_INTERPRETER_PATH_ONC_KEYS: FrozenSet[str] = frozenset(
    {
        "pathology",
        "cytology",
        "immunohistochem",
        "mammography",
        "oral_cancer",
        "dermatology",
        "wound_care",
        "surgical_specimen",
        "genetics_karyotype",
    }
)

# General X-ray / Qwen-max style (subset per product table)
_INTERPRETER_GENERAL_XRAY_KEYS: FrozenSet[str] = frozenset(
    {
        "xray_skull",
        "xray_dental",
        "spect",
        "ortho_implant",
    }
)

_REPORTS_GROUP_KEYS: FrozenSet[str] = frozenset(
    {
        "lab_report",
        "blood_report",
        "urine_report",
        "culture_report",
        "biopsy_report",
        "genetic_report",
        "radiology_report",
        "discharge_summary",
        "prescription_ocr",
        "surgical_notes",
        "spirometry",
        "nerve_conduction",
    }
)


def _interpreter_role(modality_key: str, group: str) -> str:
    if group == "reports" or modality_key in _REPORTS_GROUP_KEYS:
        return "interpreter_reports"
    if modality_key in _INTERPRETER_CARDIAC_KEYS:
        return "interpreter_cardiac"
    if modality_key in _INTERPRETER_PATH_ONC_KEYS or group in ("pathology", "oncology"):
        return "interpreter_pathology_oncology"
    if modality_key in _INTERPRETER_GENERAL_XRAY_KEYS:
        return "interpreter_general_xray"
    if group == "xray":
        return "interpreter_xray"
    return "interpreter_complex_imaging"


@dataclass(frozen=True)
class ModalityConfig:
    key: str
    group: str
    display_name: str
    input_formats: Tuple[str, ...]
    interrogator_role: str
    interpreter_role: str
    interrogator_prompt_key: str
    interpreter_prompt_key: str


def _row(
    key: str,
    group: str,
    display_name: str,
    formats: Tuple[str, ...],
) -> ModalityConfig:
    ig = INTERROGATOR_ROLE_BY_GROUP[group]
    ir = _interpreter_role(key, group)
    return ModalityConfig(
        key=key,
        group=group,
        display_name=display_name,
        input_formats=formats,
        interrogator_role=ig,
        interpreter_role=ir,
        interrogator_prompt_key=f"{group}_interrogator",
        interpreter_prompt_key=f"{group}_interpreter",
    )


# --- All 95 modalities (formats abbreviated from master reference) ---
_ROWS: List[Tuple[str, str, str, Tuple[str, ...]]] = [
    # X-Ray (11)
    ("xray", "xray", "Chest X-Ray", ("JPG", "PNG", "WEBP", "DICOM")),
    ("xray_bone", "xray", "Bone / Skeletal", ("JPG", "PNG", "DICOM")),
    ("xray_abdomen", "xray", "Abdominal X-Ray", ("JPG", "PNG", "DICOM")),
    ("xray_spine", "xray", "Spine X-Ray", ("JPG", "PNG", "DICOM")),
    ("xray_pelvis", "xray", "Pelvis X-Ray", ("JPG", "PNG", "DICOM")),
    ("xray_skull", "xray", "Skull X-Ray", ("JPG", "PNG", "DICOM")),
    ("xray_dental", "xray", "Dental / OPG", ("JPG", "PNG", "DICOM", "BMP")),
    ("xray_hand_wrist", "xray", "Hand / Wrist", ("JPG", "PNG", "DICOM")),
    ("xray_knee", "xray", "Knee X-Ray", ("JPG", "PNG", "DICOM")),
    ("xray_shoulder", "xray", "Shoulder X-Ray", ("JPG", "PNG", "DICOM")),
    ("xray_ankle_foot", "xray", "Ankle / Foot", ("JPG", "PNG", "DICOM")),
    # CT (15)
    ("ct_abdomen", "ct", "CT Abdomen / Pelvis", ("DICOM", "JPG", "PNG", "NIfTI")),
    ("ct_chest", "ct", "CT Chest", ("DICOM", "NIfTI", "JPG", "PNG")),
    ("ct_cardiac", "ct", "CT Cardiac", ("DICOM", "MP4", "JPG")),
    ("ct_spine", "ct", "CT Spine / Neuro", ("DICOM", "NIfTI", "JPG")),
    ("ct_brain", "ct", "CT Brain (NCCT)", ("DICOM", "NIfTI", "JPG", "PNG")),
    ("ct_angiography", "ct", "CT Angiography (CTA)", ("DICOM", "NIfTI", "MP4")),
    ("ct_pulmonary_pe", "ct", "CTPA — PE Detection", ("DICOM", "NIfTI", "JPG")),
    ("ct_liver", "ct", "CT Liver / Hepatic", ("DICOM", "NIfTI", "JPG")),
    ("ct_kidney", "ct", "CT KUB", ("DICOM", "JPG", "PNG")),
    ("ct_neck", "ct", "CT Neck / Soft Tissue", ("DICOM", "JPG", "PNG")),
    ("ct_sinuses", "ct", "CT Paranasal Sinuses", ("DICOM", "JPG", "PNG")),
    ("ct_whole_body", "ct", "CT Whole Body", ("DICOM", "NIfTI", "MP4")),
    ("ct_pet_fusion", "ct", "CT PET Fusion", ("DICOM", "NIfTI", "JPG")),
    ("ct_dual_energy", "ct", "CT Dual Energy", ("DICOM", "NIfTI")),
    ("ct_perfusion", "ct", "CT Perfusion Brain", ("DICOM", "MP4")),
    # MRI (14)
    ("brain_mri", "mri", "Brain MRI", ("DICOM", "NIfTI", "JPG", "PNG")),
    ("spine_mri", "mri", "Spine / Neuro MRI", ("DICOM", "NIfTI", "JPG")),
    ("cardiac_mri", "mri", "Cardiac MRI", ("DICOM", "MP4", "NIfTI")),
    ("breast_mri", "mri", "Breast MRI", ("DICOM", "NIfTI", "JPG")),
    ("liver_mri", "mri", "MRI Liver / MRCP", ("DICOM", "NIfTI", "JPG")),
    ("prostate_mri", "mri", "Prostate MRI (mpMRI)", ("DICOM", "NIfTI", "JPG")),
    ("knee_mri", "mri", "Knee MRI", ("DICOM", "NIfTI", "JPG")),
    ("shoulder_mri", "mri", "Shoulder MRI", ("DICOM", "NIfTI", "JPG")),
    ("abdomen_mri", "mri", "MRI Abdomen", ("DICOM", "NIfTI", "JPG")),
    ("fetal_mri", "mri", "Fetal / Obstetric MRI", ("DICOM", "NIfTI", "JPG")),
    ("mri_perfusion", "mri", "MRI Perfusion / DWI", ("DICOM", "NIfTI", "JPG")),
    ("mri_spectroscopy", "mri", "MRI Spectroscopy", ("DICOM", "CSV", "JPG")),
    ("wbmri", "mri", "Whole Body MRI", ("DICOM", "NIfTI", "MP4")),
    ("mri_pet_fusion", "mri", "MRI PET Fusion", ("DICOM", "NIfTI", "JPG")),
    # Nuclear (7)
    ("pet_ct", "nuclear", "PET/CT", ("DICOM", "NIfTI", "JPG")),
    ("pet_mri", "nuclear", "PET/MRI", ("DICOM", "NIfTI", "JPG")),
    ("spect", "nuclear", "SPECT Scan", ("DICOM", "JPG", "PNG")),
    ("bone_scan", "nuclear", "Bone Scintigraphy", ("DICOM", "JPG", "PNG")),
    ("thyroid_scan", "nuclear", "Thyroid Nuclear Scan", ("DICOM", "JPG", "PNG")),
    ("renal_scan", "nuclear", "Renal Scintigraphy", ("DICOM", "JPG", "PNG")),
    ("myocardial_perf", "nuclear", "Myocardial Perfusion (MPI)", ("DICOM", "MP4", "JPG")),
    # Ultrasound (12)
    ("ultrasound", "ultrasound", "General Ultrasound", ("JPG", "PNG", "MP4", "AVI")),
    ("echo_cardiac", "ultrasound", "Echocardiography", ("MP4", "AVI", "DICOM", "JPG")),
    ("us_abdomen", "ultrasound", "USG Abdomen", ("JPG", "PNG", "MP4", "DICOM")),
    ("us_pelvis", "ultrasound", "USG Pelvis", ("JPG", "PNG", "MP4", "DICOM")),
    ("us_obstetric", "ultrasound", "Obstetric USG", ("JPG", "PNG", "MP4", "DICOM")),
    ("us_thyroid", "ultrasound", "Thyroid Ultrasound", ("JPG", "PNG", "DICOM")),
    ("us_breast", "ultrasound", "Breast Ultrasound", ("JPG", "PNG", "DICOM")),
    ("us_scrotum", "ultrasound", "Scrotal Ultrasound", ("JPG", "PNG", "DICOM")),
    ("us_doppler", "ultrasound", "Doppler Ultrasound", ("JPG", "PNG", "MP4", "DICOM")),
    ("us_musculo", "ultrasound", "MSK Ultrasound", ("JPG", "PNG", "DICOM")),
    ("us_guided_biopsy", "ultrasound", "USG Biopsy / FNAC", ("JPG", "PNG", "MP4", "DICOM")),
    ("us_carotid", "ultrasound", "Carotid Doppler", ("JPG", "PNG", "MP4")),
    # Cardiac / Functional (4)
    ("ecg", "cardiac_functional", "ECG / 12-lead", ("JPG", "PNG", "PDF", "CSV")),
    ("holter_monitor", "cardiac_functional", "Holter 24hr ECG", ("PDF", "CSV", "JPG", "PNG")),
    ("spirometry", "cardiac_functional", "Pulmonary Function (PFT)", ("PDF", "PNG", "CSV", "JPG")),
    ("nerve_conduction", "cardiac_functional", "NCS / EMG", ("PDF", "JPG", "PNG", "CSV")),
    # Specialized (10)
    ("fluoroscopy", "specialized", "Fluoroscopy / Barium", ("MP4", "AVI", "JPG", "DICOM")),
    ("angiography", "specialized", "Conventional DSA", ("DICOM", "MP4", "JPG")),
    ("dexa_scan", "specialized", "DEXA Bone Density", ("JPG", "PNG", "PDF", "DICOM")),
    ("endoscopy", "specialized", "Endoscopy / Colonoscopy", ("JPG", "PNG", "MP4", "AVI")),
    ("bronchoscopy", "specialized", "Bronchoscopy", ("JPG", "PNG", "MP4")),
    ("oct_retinal", "specialized", "OCT Retinal Scan", ("JPG", "PNG", "DICOM")),
    ("fundus_photo", "specialized", "Fundus Photography", ("JPG", "PNG", "TIFF")),
    ("colposcopy", "specialized", "Colposcopy", ("JPG", "PNG", "MP4")),
    ("thermography", "specialized", "Medical Thermography", ("JPG", "PNG", "TIFF")),
    ("capsule_endoscopy", "specialized", "Capsule Endoscopy", ("JPG", "PNG", "MP4")),
    # Pathology (5)
    ("pathology", "pathology", "Pathology Slides", ("JPG", "PNG", "TIFF", "SVS")),
    ("cytology", "pathology", "Cytology Slides", ("JPG", "PNG", "TIFF")),
    ("immunohistochem", "pathology", "IHC Slides", ("JPG", "PNG", "TIFF", "SVS")),
    ("surgical_specimen", "pathology", "Surgical Specimen", ("JPG", "PNG")),
    ("genetics_karyotype", "pathology", "Karyotype Analysis", ("JPG", "PNG", "TIFF")),
    # Oncology (4)
    ("mammography", "oncology", "Mammography", ("DICOM", "JPG", "PNG")),
    ("oral_cancer", "oncology", "Oral Cancer Imaging", ("JPG", "PNG", "MP4")),
    ("dermatology", "oncology", "Dermatology", ("JPG", "PNG", "TIFF")),
    ("wound_care", "oncology", "Wound / Ulcer", ("JPG", "PNG", "MP4")),
    # Ophthalmology / Dental (3)
    ("ophthalmology", "ophthalmology_dental", "Slit Lamp / Anterior Segment", ("JPG", "PNG", "MP4")),
    ("dental_cbct", "ophthalmology_dental", "Dental CBCT / 3D", ("DICOM", "JPG", "PNG", "STL")),
    ("ortho_implant", "ophthalmology_dental", "Post-Op Implant X-Ray", ("JPG", "PNG", "DICOM")),
    # Reports (10)
    ("lab_report", "reports", "Lab Reports", ("PDF", "JPG", "PNG", "TXT", "CSV")),
    ("blood_report", "reports", "CBC / Blood Report", ("PDF", "JPG", "PNG", "CSV")),
    ("urine_report", "reports", "Urine Analysis", ("PDF", "JPG", "PNG", "TXT")),
    ("culture_report", "reports", "Microbiology Culture", ("PDF", "JPG", "PNG", "TXT")),
    ("biopsy_report", "reports", "Biopsy / Histopathology", ("PDF", "JPG", "PNG", "TXT")),
    ("genetic_report", "reports", "Genetic / DNA Report", ("PDF", "TXT", "CSV", "JSON")),
    ("radiology_report", "reports", "Radiology Report Text", ("PDF", "TXT", "DOCX")),
    ("discharge_summary", "reports", "Discharge Summary", ("PDF", "TXT", "JPG", "DOCX")),
    ("prescription_ocr", "reports", "Prescription / Handwritten", ("JPG", "PNG", "PDF")),
    ("surgical_notes", "reports", "Operative / Surgical Notes", ("PDF", "TXT", "DOCX")),
]

MODALITY_REGISTRY: Dict[str, ModalityConfig] = {
    k: _row(k, g, dn, fmt) for k, g, dn, fmt in _ROWS
}

ALL_MODALITY_KEYS: List[str] = [k for k, _, _, _ in _ROWS]
AUTO_DETECT_KEY = "auto"


def get_modality_config(key: str) -> Optional[ModalityConfig]:
    k = (key or "").strip().lower()
    if k == AUTO_DETECT_KEY:
        return None
    return MODALITY_REGISTRY.get(k)


def list_modalities_for_prompt() -> str:
    """Compact list for auto-detect system prompt."""
    lines = []
    for k, g, dn, _ in _ROWS:
        lines.append(f"- {k}: {dn} (group: {g})")
    return "\n".join(lines)


def validate_modality_key(key: str) -> bool:
    k = (key or "").strip().lower()
    return k == AUTO_DETECT_KEY or k in MODALITY_REGISTRY
