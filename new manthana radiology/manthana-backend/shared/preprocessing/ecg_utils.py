"""
Manthana — ECG Utilities
Parse CSV, EDF, DICOM-ECG signal data for the ECG service.
"""

import os
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger("manthana.ecg_utils")

# Standard 12-lead ECG channel names
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def read_ecg_csv(filepath: str, sample_rate: int = 500) -> Tuple[np.ndarray, int]:
    """Read ECG data from CSV file.
    
    Expects columns for each lead (12 columns) or a single column
    with lead data interleaved.
    
    Returns:
        (signal_array, sample_rate)
        signal_array shape: (num_leads, num_samples)
    """
    import pandas as pd

    df = pd.read_csv(filepath)
    
    # Try to detect format
    if len(df.columns) >= 12:
        # 12+ columns: each column is a lead
        signal = df.iloc[:, :12].values.T.astype(np.float32)
    elif len(df.columns) == 1:
        # Single column: assume interleaved 12-lead
        data = df.iloc[:, 0].values.astype(np.float32)
        num_samples = len(data) // 12
        signal = data[:num_samples * 12].reshape(12, num_samples)
    else:
        # Try with whatever columns we have
        signal = df.values.T.astype(np.float32)

    logger.info(f"Loaded ECG CSV: {signal.shape[0]} leads, "
                f"{signal.shape[1]} samples at {sample_rate} Hz")
    return signal, sample_rate


def read_ecg_edf(filepath: str) -> Tuple[np.ndarray, int]:
    """Read ECG data from EDF (European Data Format) file.
    
    Returns:
        (signal_array, sample_rate)
    """
    import pyedflib

    f = pyedflib.EdfReader(filepath)
    
    n_channels = f.signals_in_file
    sample_rate = int(f.getSampleFrequency(0))
    
    signals = []
    for i in range(min(n_channels, 12)):
        signals.append(f.readSignal(i))
    
    f.close()
    
    signal = np.array(signals, dtype=np.float32)
    
    logger.info(f"Loaded ECG EDF: {signal.shape[0]} leads, "
                f"{signal.shape[1]} samples at {sample_rate} Hz")
    return signal, sample_rate


def read_ecg_dicom(filepath: str) -> Tuple[np.ndarray, int]:
    """Read ECG data from DICOM-ECG file.
    
    Returns:
        (signal_array, sample_rate)
    """
    import pydicom

    ds = pydicom.dcmread(filepath)
    
    # DICOM waveform data
    waveform_seq = ds.WaveformSequence[0]
    channels = waveform_seq.NumberOfWaveformChannels
    samples = waveform_seq.NumberOfWaveformSamples
    sample_rate = int(waveform_seq.SamplingFrequency)
    
    # Extract waveform data
    data = np.frombuffer(
        waveform_seq.WaveformData, 
        dtype=np.int16
    ).reshape(samples, channels).T.astype(np.float32)

    # Apply channel sensitivity
    for i, ch_def in enumerate(waveform_seq.ChannelDefinitionSequence):
        sensitivity = float(getattr(ch_def, "ChannelSensitivity", 1.0))
        correction = float(getattr(ch_def, "ChannelSensitivityCorrectionFactor", 1.0))
        baseline = float(getattr(ch_def, "ChannelBaseline", 0.0))
        data[i] = (data[i] + baseline) * sensitivity * correction

    logger.info(f"Loaded DICOM-ECG: {channels} leads, "
                f"{samples} samples at {sample_rate} Hz")
    return data[:12], sample_rate


def normalize_ecg(signal: np.ndarray, target_rate: int = 500,
                  current_rate: int = None) -> np.ndarray:
    """Normalize ECG signal for model input.
    
    Steps:
    1. Resample to target_rate if needed
    2. Remove baseline wander
    3. Normalize amplitude to [-1, 1]
    """
    from scipy import signal as scipy_signal

    # Resample if needed
    if current_rate and current_rate != target_rate:
        num_samples = int(signal.shape[1] * target_rate / current_rate)
        resampled = np.zeros((signal.shape[0], num_samples), dtype=np.float32)
        for i in range(signal.shape[0]):
            resampled[i] = scipy_signal.resample(signal[i], num_samples)
        signal = resampled

    # Remove baseline wander (high-pass filter at 0.5 Hz)
    sos = scipy_signal.butter(4, 0.5, btype="high", fs=target_rate, output="sos")
    for i in range(signal.shape[0]):
        signal[i] = scipy_signal.sosfiltfilt(sos, signal[i])

    # Normalize each lead to [-1, 1]
    for i in range(signal.shape[0]):
        max_val = np.max(np.abs(signal[i]))
        if max_val > 0:
            signal[i] = signal[i] / max_val

    return signal


def detect_input_type(filepath: str) -> str:
    """Detect the type of ECG input file.
    
    Returns: "csv", "edf", "dicom", "image", "pdf_rejected", or "unknown"
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in (".csv", ".tsv", ".txt"):
        return "csv"
    elif ext in (".edf", ".bdf"):
        return "edf"
    elif ext in (".dcm",):
        return "dicom"
    elif ext == ".pdf":
        return "pdf_rejected"
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        return "image"
    else:
        # Try to detect by content
        try:
            import pydicom
            pydicom.dcmread(filepath, stop_before_pixels=True)
            return "dicom"
        except Exception:
            pass
        return "unknown"
