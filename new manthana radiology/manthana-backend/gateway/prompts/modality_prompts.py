"""Per-group interpreter specialization (appended to base interpreter prompt)."""

_GROUP_SPECIALIZATION: dict[str, str] = {
    "xray": (
        "Consider positioning, AP vs PA, lines/tubes, cardiothoracic ratio. "
        "In India, consider TB and post-primary sequelae in differential for cavitary or nodular patterns."
    ),
    "ct": (
        "Report by relevant windows (lung / mediastinal / bone). Note contrast phase if inferable. "
        "Structured organ-by-organ findings when appropriate."
    ),
    "mri": (
        "Reference sequences if visible (T1/T2/FLAIR/DWI). Describe signal, enhancement pattern, "
        "and location precisely."
    ),
    "nuclear": (
        "Describe uptake pattern; distinguish physiologic vs pathologic when possible; mention SUV if visible."
    ),
    "ultrasound": (
        "Describe echogenicity, size, vascularity; use standard scoring (e.g. TIRADS/BIRADS/OB) when applicable."
    ),
    "cardiac_functional": (
        "For ECG/functional: systematic rate, rhythm, axis, intervals, ST/T, chamber clues. "
        "For PFT/EMG: summarize key numeric and pattern findings without replacing device interpretation."
    ),
    "specialized": (
        "Use modality-appropriate structure (endoscopy mucosa, OCT layers, DEXA T-scores, etc.)."
    ),
    "pathology": (
        "Architecture, cytology, mitoses, margins; WHO-style terminology where appropriate; "
        "IHC as described."
    ),
    "oncology": (
        "Screening/staging language; avoid over-staging from imaging alone; recommend confirmatory steps."
    ),
    "ophthalmology_dental": (
        "Anterior segment / fundus / dental CBCT as applicable; implants: hardware and position."
    ),
    "reports": (
        "Extract structured values; flag critical/panic labs; compare to reference ranges; trends if present."
    ),
}


def group_specialization_for(group: str) -> str:
    return _GROUP_SPECIALIZATION.get(group, _GROUP_SPECIALIZATION["specialized"])
