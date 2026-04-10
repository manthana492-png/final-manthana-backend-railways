"""
Manthana — Service Router
Maps modality names to internal Docker service URLs.
"""

import os
from fastapi import HTTPException


def _service_url(env_key: str, default: str) -> str:
    raw = (os.getenv(env_key) or "").strip()
    return raw if raw else default


SERVICE_MAP = {
    "xray": _service_url(
        "XRAY_SERVICE_URL",
        "http://body_xray:8001/analyze/xray",
    ),
    "brain_mri": _service_url(
        "BRAIN_MRI_SERVICE_URL",
        "http://brain_mri:8002/analyze/brain_mri",
    ),
    "cardiac_ct": _service_url(
        "CARDIAC_CT_SERVICE_URL",
        "http://cardiac_ct:8004/analyze/cardiac_ct",
    ),
    "pathology": _service_url(
        "PATHOLOGY_SERVICE_URL",
        "http://pathology:8005/analyze/pathology",
    ),
    "abdominal_ct": _service_url(
        "ABDOMINAL_CT_SERVICE_URL",
        "http://abdominal_ct:8008/analyze/abdominal_ct",
    ),
    "ultrasound": _service_url(
        "ULTRASOUND_SERVICE_URL",
        "http://ultrasound:8009/analyze/ultrasound",
    ),
    "spine_neuro": _service_url(
        "SPINE_NEURO_SERVICE_URL",
        "http://spine_neuro:8010/analyze/spine_neuro",
    ),
    "ct_brain": _service_url(
        "CT_BRAIN_SERVICE_URL",
        "http://ct_brain:8017/analyze/ct_brain",
    ),
    "ct_brain_vista": _service_url(
        "CT_BRAIN_VISTA_SERVICE_URL",
        "http://ct_brain:8017/analyze/ct_brain",
    ),
    "cytology": _service_url(
        "CYTOLOGY_SERVICE_URL",
        "http://cytology:8011/analyze/cytology",
    ),
    "mammography": _service_url(
        "MAMMOGRAPHY_SERVICE_URL",
        "http://mammography:8012/analyze/mammography",
    ),
    "ecg": _service_url(
        "ECG_SERVICE_URL",
        "http://ecg:8013/analyze/ecg",
    ),
    # Production: point ORAL_CANCER_SERVICE_URL at Modal GPU manthana-oral-cancer …/analyze/oral_cancer
    "oral_cancer": _service_url(
        "ORAL_CANCER_SERVICE_URL",
        "http://oral_cancer:8014/analyze/oral_cancer",
    ),
    "lab_report": _service_url(
        "LAB_REPORT_SERVICE_URL",
        "http://lab_report:8015/analyze/lab_report",
    ),
    "dermatology": _service_url(
        "DERMATOLOGY_SERVICE_URL",
        "http://dermatology:8016/analyze/dermatology",
    ),
    "pacs": "http://pacs_bridge:8030/pacs",
}

# Aliases — user-friendly names that map to the same service
ALIASES = {
    "chest_xray": "xray",
    "chest": "xray",
    "bone": "xray",
    "fracture": "xray",
    "x-ray": "xray",
    "mri": "brain_mri",           # Unified MRI → gateway auto-detects region
    "brain": "brain_mri",
    "head_mri": "brain_mri",
    "heart": "cardiac_ct",
    "cardiac": "cardiac_ct",
    "ct": "abdominal_ct",          # Legacy unified CT alias; clients should prefer abdominal_ct / chest_ct / explicit UI subtypes
    "ct_scan": "abdominal_ct",
    "chest_ct": "abdominal_ct",  # Thoracic CT — same TotalSeg pipeline; ct_region in patient_context
    "spine_ct": "spine_neuro",
    "spine_mri": "spine_neuro",
    "mr_spine": "spine_neuro",
    "brain_ct": "ct_brain",
    "head_ct": "ct_brain",
    "ncct_brain": "ct_brain",
    "patho": "pathology",
    "wsi": "pathology",
    # Panoramic / dental radiographs → general X-ray pipeline (no dedicated OPG service)
    "dental": "xray",
    "opg": "xray",
    "abdomen": "abdominal_ct",
    "us": "ultrasound",
    "usg": "ultrasound",
    "spine": "spine_neuro",
    "neuro": "spine_neuro",
    "cyto": "cytology",
    "pap": "cytology",
    "mammo": "mammography",
    "breast": "mammography",
    "mouth": "oral_cancer",
    "oral": "oral_cancer",
    "lab": "lab_report",
    "labs": "lab_report",
    "blood_test": "lab_report",
    "blood": "lab_report",
    "urine": "lab_report",
    "biochemistry": "lab_report",
    "dicom": "pacs",
    "skin": "dermatology",
    "derm": "dermatology",
    "skin_lesion": "dermatology",
    "vista_ct": "ct_brain_vista",
    "ct_brain_premium": "ct_brain_vista",
}


def route_to_service(modality: str) -> str:
    """Get the internal service URL for a given modality.
    
    Supports both canonical names and user-friendly aliases.
    
    Raises:
        ValueError: If modality is not recognized
    """
    modality = modality.lower().strip()

    # Check direct match
    if modality in SERVICE_MAP:
        return SERVICE_MAP[modality]

    # Check aliases
    if modality in ALIASES:
        canonical = ALIASES[modality]
        return SERVICE_MAP[canonical]

    available = sorted(list(SERVICE_MAP.keys()) + list(ALIASES.keys()))
    raise HTTPException(
        status_code=422,
        detail=(
            f"Unknown modality: '{modality}'. "
            f"Available: {', '.join(sorted(SERVICE_MAP.keys()))}"
        ),
    )


def get_all_modalities() -> list:
    """Return list of all supported modalities."""
    return sorted(SERVICE_MAP.keys())
