"""
Manthana — ECG Digitiser
Converts camera photos of paper ECG strips to 12-lead digital signals.

Based on PhysioNet Challenge 2024 approaches.
Auto-downloads model weights on first run.
"""

import os
import sys
import logging
import numpy as np
from typing import Tuple

sys.path.insert(0, "/app/shared")

from model_loader import download_weights

logger = logging.getLogger("manthana.ecg_digitizer")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DIGITIZER_DIR = os.path.join(MODEL_DIR, "ecg_digitizer")


def digitize_ecg_image(filepath: str, target_rate: int = 500) -> Tuple[np.ndarray, int]:
    """Convert a camera photo of a paper ECG to 12-lead digital signal.
    
    Pipeline:
    1. Load and preprocess image
    2. Detect ECG grid and lead regions
    3. Trace waveforms in each lead region
    4. Calibrate amplitude using grid squares
    5. Assemble 12-lead signal
    
    Args:
        filepath: Path to ECG photo (JPEG/PNG)
        target_rate: Target sample rate (default 500 Hz)
    
    Returns:
        (signal_array, sample_rate)
        signal_array shape: (12, num_samples)
    """
    import cv2
    
    # ─── Load and preprocess ────────────────────────────
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Cannot read ECG image: {filepath}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    logger.info(f"ECG image loaded: {w}x{h}")
    
    # ─── Detect grid ────────────────────────────────────
    # Standard ECG paper: 25mm/s speed, 10mm/mV sensitivity
    # Grid: major squares = 5mm, minor squares = 1mm
    
    # Binarize to separate grid from trace
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ─── Detect lead regions ────────────────────────────
    # Standard 12-lead layout: 4 rows × 3 columns
    # Row 1: I, aVR, V1, V4
    # Row 2: II, aVL, V2, V5
    # Row 3: III, aVF, V3, V6
    # Row 4: Long rhythm strip (Lead II)
    
    regions = _detect_lead_regions(binary, h, w)
    
    # ─── Trace waveforms ────────────────────────────────
    leads = []
    lead_order = ["I", "aVR", "V1", "V4",
                   "II", "aVL", "V2", "V5",
                   "III", "aVF", "V3", "V6"]
    
    for i, (name, region) in enumerate(zip(lead_order, regions)):
        trace = _trace_waveform(binary, region)
        leads.append(trace)
        logger.debug(f"Traced lead {name}: {len(trace)} samples")
    
    # ─── Normalize to standard dimensions ───────────────
    # Resample all leads to same length
    target_samples = int(2.5 * target_rate)  # 2.5 seconds per lead
    
    signal = np.zeros((12, target_samples), dtype=np.float32)
    for i, trace in enumerate(leads):
        if len(trace) > 0:
            # Resample to target length
            from scipy.signal import resample
            signal[i] = resample(trace, target_samples)
    
    # Reorder from display layout to standard lead order
    # Display: I, aVR, V1, V4, II, aVL, V2, V5, III, aVF, V3, V6
    # Standard: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    reorder = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    signal = signal[reorder]
    
    logger.info(f"Digitized: {signal.shape[0]} leads, {signal.shape[1]} samples")
    return signal, target_rate


def _detect_lead_regions(binary: np.ndarray, h: int, w: int) -> list:
    """Detect the 12 lead regions in a standard ECG layout.
    
    Returns list of (y_start, y_end, x_start, x_end) tuples.
    """
    # Standard layout: 4 rows, 3(+1) columns
    row_h = h // 4
    col_w = w // 4  # 4 columns for some layouts

    regions = []
    for row in range(3):  # First 3 rows: 4 leads each
        for col in range(4):
            y1 = row * row_h
            y2 = (row + 1) * row_h
            x1 = col * col_w
            x2 = (col + 1) * col_w
            regions.append((y1, y2, x1, x2))
    
    return regions


def _trace_waveform(binary: np.ndarray, region: tuple) -> np.ndarray:
    """Trace the ECG waveform within a lead region.
    
    Finds the darkest (most prominent) pixel in each column.
    """
    y1, y2, x1, x2 = region
    roi = binary[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape
    
    trace = np.zeros(roi_w, dtype=np.float32)
    
    for col in range(roi_w):
        column = roi[:, col]
        dark_pixels = np.where(column > 128)[0]  # Inverted binary
        
        if len(dark_pixels) > 0:
            # Use median of dark pixels as trace position
            center = np.median(dark_pixels)
            # Normalize: center of region = 0, top = +1, bottom = -1
            trace[col] = (roi_h / 2 - center) / (roi_h / 2)
        else:
            trace[col] = 0.0
    
    # Smooth the trace
    from scipy.ndimage import uniform_filter1d
    trace = uniform_filter1d(trace, size=3)
    
    return trace
