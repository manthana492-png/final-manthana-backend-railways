"""
Shared DICOM utilities across CT/MRI services.
"""

import os
from pathlib import Path


def count_dicoms_in_tree(root: str) -> int:
    """Count .dcm and .dic files in directory tree."""
    n = 0
    for _, _, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if fl.endswith(".dcm") or fl.endswith(".dic"):
                n += 1
    return n


def find_dicom_files(directory: str) -> list[str]:
    """Find all DICOM file paths in directory, sorted."""
    paths = []
    p = Path(directory)
    for ext in (".dcm", ".dic", ".dicom"):
        paths.extend(p.glob(f"*{ext}"))
        paths.extend(p.glob(f"*{ext.upper()}"))
    # Also include files without extension that might be DICOM
    for f in p.iterdir():
        if f.is_file() and f.suffix == "":
            paths.append(f)
    return sorted([str(x) for x in paths if x.is_file()])