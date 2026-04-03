"""
DICOM Modality Router — auto-detect Manthana service from DICOM tags.

Mammography (MG): routing returns modality mammography only. A separate ingest/queue
step should populate patient_context_json.views with four server paths (L-CC, L-MLO,
R-CC, R-MLO) when a full exam is available — this module does not build that dict.
"""


# Modality mapping: DICOM Modality tag → Manthana service
MODALITY_MAP = {
    "CR": "xray",       # Computed Radiography
    "DX": "xray",       # Digital Radiography
    "RF": "xray",       # Radiofluoroscopy
    "MR": "brain_mri",  # Magnetic Resonance (default → brain)
    "CT": "abdominal_ct",  # Computed Tomography (default → abdominal)
    "US": "ultrasound",
    "MG": "mammography",
    "PT": "xray",       # PET → fallback
    "NM": "xray",       # Nuclear Medicine → fallback
    "OPG": "xray",
    "IO": "xray",  # Intra-oral
    "PX": "xray",  # Panoramic X-ray
    "ECG": "ecg",
    "HD": "ecg",         # Hemodynamic → ECG
    "ES": "ultrasound",  # Endoscopy
    "XA": "cardiac_ct",  # X-ray Angiography → cardiac
    "XC": "pathology",   # External Camera Photography
    "SM": "pathology",   # Slide Microscopy
}

# Body part refinement for CT and MR
BODY_PART_CT = {
    "CHEST": "cardiac_ct",
    "HEART": "cardiac_ct",
    "CARDIAC": "cardiac_ct",
    "THORAX": "cardiac_ct",
    "ABDOMEN": "abdominal_ct",
    "PELVIS": "abdominal_ct",
    "LIVER": "abdominal_ct",
    "KIDNEY": "abdominal_ct",
    "SPINE": "spine_neuro",
    "LSPINE": "spine_neuro",
    "CSPINE": "spine_neuro",
    "TSPINE": "spine_neuro",
    "HEAD": "brain_mri",
    "BRAIN": "brain_mri",
    "SKULL": "brain_mri",
    "NECK": "spine_neuro",
}

BODY_PART_MR = {
    "BRAIN": "brain_mri",
    "HEAD": "brain_mri",
    "SKULL": "brain_mri",
    "SPINE": "spine_neuro",
    "LSPINE": "spine_neuro",
    "CSPINE": "spine_neuro",
    "TSPINE": "spine_neuro",
    "NECK": "spine_neuro",
    "KNEE": "unsupported_mr_msk",
    "SHOULDER": "unsupported_mr_msk",
    "ANKLE": "unsupported_mr_msk",
    "WRIST": "unsupported_mr_msk",
    "ELBOW": "unsupported_mr_msk",
    "HIP": "unsupported_mr_msk",
    "FOOT": "unsupported_mr_msk",
    "ABDOMEN": "abdominal_ct",
    "PELVIS": "abdominal_ct",
    "BREAST": "mammography",
    "CARDIAC": "cardiac_ct",
    "HEART": "cardiac_ct",
}


def detect_modality_from_tags(main_tags: dict, patient_tags: dict = None) -> str:
    """
    Detect Manthana service modality from DICOM study tags.

    Args:
        main_tags: Study MainDicomTags from Orthanc
        patient_tags: Patient MainDicomTags from Orthanc

    Returns:
        Manthana modality string (e.g. "xray", "brain_mri", "cardiac_ct")
    """
    modality = (main_tags.get("ModalitiesInStudy", "") or
                main_tags.get("Modality", "")).strip().upper()
    body_part = (main_tags.get("BodyPartExamined", "") or "").strip().upper()
    study_desc = (main_tags.get("StudyDescription", "") or "").strip().upper()
    series_desc = (main_tags.get("SeriesDescription", "") or "").strip().upper()

    # Handle multiple modalities (e.g. "CT\\PT" for PET-CT)
    if "\\" in modality:
        modalities = modality.split("\\")
        # Prefer CT over PT, MR over secondary
        for m in ["CT", "MR", "CR", "DX", "US", "MG"]:
            if m in modalities:
                modality = m
                break
        else:
            modality = modalities[0]

    # Base modality lookup
    service = MODALITY_MAP.get(modality, "xray")

    # Refine CT by body part
    if modality == "CT":
        for keyword, svc in BODY_PART_CT.items():
            if keyword in body_part or keyword in study_desc:
                service = svc
                break

    # Refine MR by body part
    elif modality in ("MR", "MRI"):
        for keyword, svc in BODY_PART_MR.items():
            if keyword in body_part or keyword in study_desc:
                service = svc
                break

    # Study description fallback
    if service == "xray":
        desc = study_desc + " " + series_desc
        if "MAMMO" in desc or "BREAST" in desc:
            service = "mammography"
        elif "DENTAL" in desc or "OPG" in desc:
            service = "xray"
        elif "ECG" in desc or "EKG" in desc:
            service = "ecg"
        elif "ORAL" in desc or "MOUTH" in desc:
            service = "oral_cancer"

    return service
