"""
Shared upload preparation for imaging services (ZIP extract, single DICOM in folder).

Used by abdominal CT, CT brain, brain MRI, cardiac CT, spine/neuro.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile

from fastapi import HTTPException

logger = logging.getLogger("manthana.service_upload_prep")


def prepare_upload_for_pipeline(
    saved_path: str,
    filename: str | None,
    job_id: str,
) -> tuple[str, list[str]]:
    """
    Return (path_for_run_pipeline, cleanup_dir_paths).

    - ZIP → extract to temp dir (DICOM series or film-photo PNG/JPEG batch).
    - Single .dcm/.dic → temp dir with one instance (TotalSegmentator expects a folder).
    - Raster / NIfTI → use saved file path as-is.
    """
    cleanup: list[str] = []
    ext = (os.path.splitext(filename or "")[1] or "").lower()
    lower_name = (filename or "").lower()

    if ext == ".zip" or lower_name.endswith(".zip"):
        d = tempfile.mkdtemp(prefix=f"upload_zip_{job_id}_")
        cleanup.append(d)
        try:
            with zipfile.ZipFile(saved_path, "r") as zf:
                zf.extractall(d)
        except zipfile.BadZipFile as e:
            shutil.rmtree(d, ignore_errors=True)
            cleanup.clear()
            raise HTTPException(status_code=400, detail=f"Invalid ZIP: {e}") from e
        return d, cleanup

    if ext in (".dcm", ".dic") or lower_name.endswith((".dcm", ".dic")):
        d = tempfile.mkdtemp(prefix=f"upload_series_{job_id}_")
        cleanup.append(d)
        dest_name = os.path.basename(filename) if filename else f"instance{ext or '.dcm'}"
        if not dest_name.lower().endswith((".dcm", ".dic")):
            dest_name = "instance.dcm"
        shutil.copy2(saved_path, os.path.join(d, dest_name))
        return d, cleanup

    return saved_path, cleanup
