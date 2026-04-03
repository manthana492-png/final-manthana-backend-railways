"""Detect MR vs CT from optional gateway hint or first DICOM in series_dir."""

from __future__ import annotations

import os
from pathlib import Path


def is_mr_input(source_modality: str | None, series_dir: str | None) -> bool:
    """True if explicit MR or first DICOM in series has Modality MR."""
    if (source_modality or "").upper().strip() == "MR":
        return True
    if not series_dir or not os.path.isdir(series_dir):
        return False
    try:
        import pydicom
    except ImportError:
        return False
    for f in sorted(Path(series_dir).iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            mod = (getattr(ds, "Modality", "") or "").upper()
            if mod == "MR":
                return True
            if mod == "CT":
                return False
        except Exception:
            continue
    return False
