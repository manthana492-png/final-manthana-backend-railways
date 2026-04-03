"""
Manthana — DICOM Utilities
Read DICOM files, extract pixel data, convert series to NIfTI.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("manthana.dicom_utils")


def read_dicom(filepath: str) -> Tuple[np.ndarray, dict]:
    """Read a single DICOM file, return pixel array and metadata.
    
    Returns:
        (pixel_array, metadata_dict)
    """
    import pydicom

    ds = pydicom.dcmread(filepath)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale if present
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    pixel_array = pixel_array * float(slope) + float(intercept)

    metadata = {
        "patient_id": getattr(ds, "PatientID", ""),
        "study_description": getattr(ds, "StudyDescription", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
        "modality": getattr(ds, "Modality", ""),
        "body_part": getattr(ds, "BodyPartExamined", ""),
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "pixel_spacing": list(getattr(ds, "PixelSpacing", [1.0, 1.0])),
        "slice_thickness": float(getattr(ds, "SliceThickness", 1.0)),
    }

    return pixel_array, metadata


def read_dicom_series(directory: str) -> Tuple[np.ndarray, dict]:
    """Read a directory of DICOM files as a 3D volume.
    
    Sorts slices by ImagePositionPatient or InstanceNumber.
    
    Returns:
        (3d_volume_array, metadata_dict)
    """
    import pydicom

    dcm_files = []
    for f in Path(directory).glob("*"):
        if f.is_file():
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                dcm_files.append((str(f), ds))
            except Exception:
                continue

    if not dcm_files:
        raise ValueError(f"No valid DICOM files found in {directory}")

    # Sort by slice position
    def sort_key(item):
        ds = item[1]
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos:
            return float(pos[2])  # Z position
        return float(getattr(ds, "InstanceNumber", 0))

    dcm_files.sort(key=sort_key)

    # Read all slices
    slices = []
    for filepath, _ in dcm_files:
        ds = pydicom.dcmread(filepath)
        pixel = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel = pixel * slope + intercept
        slices.append(pixel)

    volume = np.stack(slices, axis=0)  # Shape: (slices, H, W)
    
    # Metadata from first slice
    first_ds = pydicom.dcmread(dcm_files[0][0])
    metadata = {
        "patient_id": getattr(first_ds, "PatientID", ""),
        "study_description": getattr(first_ds, "StudyDescription", ""),
        "modality": getattr(first_ds, "Modality", ""),
        "body_part": getattr(first_ds, "BodyPartExamined", ""),
        "num_slices": len(slices),
        "volume_shape": list(volume.shape),
        "pixel_spacing": list(getattr(first_ds, "PixelSpacing", [1.0, 1.0])),
        "slice_thickness": float(getattr(first_ds, "SliceThickness", 1.0)),
    }

    logger.info(f"Loaded DICOM series: {len(slices)} slices, shape={volume.shape}")
    return volume, metadata


def dicom_series_to_nifti(dicom_dir: str, output_path: str = None) -> str:
    """Convert a DICOM series directory to NIfTI format.
    
    Returns path to the .nii.gz file.
    """
    import nibabel as nib

    volume, metadata = read_dicom_series(dicom_dir)
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".nii.gz")

    # Create affine matrix from DICOM metadata
    spacing = metadata.get("pixel_spacing", [1.0, 1.0])
    thickness = metadata.get("slice_thickness", 1.0)
    
    affine = np.diag([
        float(spacing[0]),
        float(spacing[1]),
        float(thickness),
        1.0,
    ])

    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nii, output_path)

    logger.info(f"Converted DICOM → NIfTI: {output_path}")
    return output_path


def dicom_to_png(filepath: str, output_path: str = None) -> str:
    """Convert a single DICOM to PNG for display/preprocessing."""
    from PIL import Image

    pixel_array, _ = read_dicom(filepath)
    
    # Normalize to 0-255
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        normalized = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(pixel_array, dtype=np.uint8)

    img = Image.fromarray(normalized)
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")
    
    img.save(output_path)
    return output_path
