"""
Manthana — NIfTI Utilities
Read/write NIfTI volumes for CT and MRI processing.
"""

import logging
import tempfile
import numpy as np

logger = logging.getLogger("manthana.nifti_utils")


def read_nifti(filepath: str) -> tuple:
    """Read a NIfTI file, return volume and affine.
    
    Returns:
        (volume_array, affine_matrix)
    """
    import nibabel as nib

    nii = nib.load(filepath)
    volume = nii.get_fdata().astype(np.float32)
    affine = nii.affine

    logger.info(f"Loaded NIfTI: shape={volume.shape}, dtype={volume.dtype}")
    return volume, affine


def save_nifti(volume: np.ndarray, affine: np.ndarray, 
               output_path: str = None) -> str:
    """Save a numpy volume as NIfTI file."""
    import nibabel as nib

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".nii.gz")

    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nii, output_path)
    
    logger.info(f"Saved NIfTI: {output_path}")
    return output_path


def window_ct(volume: np.ndarray, window_center: float, 
              window_width: float) -> np.ndarray:
    """Apply CT windowing (e.g., lung window, bone window).
    
    Common windows:
        Lung: center=-600, width=1500
        Soft tissue: center=40, width=400  
        Bone: center=400, width=1800
        Brain: center=40, width=80
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed.astype(np.float32)
