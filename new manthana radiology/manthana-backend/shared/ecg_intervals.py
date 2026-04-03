"""
ECG interval measurements (HR, PR, QRS, QT, QTc) via neurokit2 on Lead II.
Peak delineation columns come from the signals DataFrame (boolean masks), not info dict.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("manthana.ecg_intervals")

# Columns verified for neurokit2 >= 0.2.7 (see plan pre-check)
_COL_PR_START = "ECG_P_Onsets"
_COL_PR_END = "ECG_Q_Peaks"
_FALLBACK_PR = ("ECG_P_Peaks", "ECG_Q_Peaks")
_COL_QRS_START = "ECG_Q_Peaks"
_COL_QRS_END = "ECG_S_Peaks"
_COL_QT_START = "ECG_Q_Peaks"
_COL_QT_END = "ECG_T_Offsets"


def _peaks_from_df(signals_df, col: str) -> np.ndarray:
    """Sample indices where boolean mask column == 1."""
    if col not in signals_df.columns:
        return np.array([], dtype=np.int64)
    s = signals_df[col].fillna(0).astype(float)
    return np.where(s >= 0.5)[0]


def _interval_ms(
    signals_df,
    start_col: str,
    end_col: str,
    sample_rate: float,
) -> Optional[float]:
    """Median interval in ms between paired start/end peak indices (aligned by order)."""
    starts = _peaks_from_df(signals_df, start_col)
    ends = _peaks_from_df(signals_df, end_col)
    if len(starts) < 2 or len(ends) < 2:
        return None
    n = min(len(starts), len(ends))
    diffs = np.abs(ends[:n].astype(np.float64) - starts[:n].astype(np.float64))
    med = float(np.median(diffs))
    return med / float(sample_rate) * 1000.0


def _bazett(qt_ms: Optional[float], hr_bpm: float) -> Optional[float]:
    if qt_ms is None or hr_bpm is None or hr_bpm <= 0:
        return None
    rr_sec = 60.0 / float(hr_bpm)
    return float(qt_ms / (rr_sec**0.5))


def _scipy_hr_fallback(signal: np.ndarray, sample_rate: float) -> dict[str, Any]:
    import scipy.signal

    if signal is None or signal.size == 0:
        return {
            "signal_quality": "insufficient",
            "hr_bpm": None,
            "pr_ms": None,
            "qrs_ms": None,
            "qt_ms": None,
            "qtc_ms": None,
            "num_leads": 0,
            "method": "no-signal",
        }

    lead = signal[1] if signal.shape[0] > 1 else signal[0]
    peaks, _ = scipy.signal.find_peaks(lead, distance=int(sample_rate * 0.4))
    if len(peaks) >= 2:
        rr = np.diff(peaks) / float(sample_rate)
        hr = float(60.0 / np.mean(rr))
    else:
        hr = 75.0
    return {
        "hr_bpm": round(hr, 1),
        "pr_ms": None,
        "qrs_ms": None,
        "qt_ms": None,
        "qtc_ms": None,
        "num_leads": int(signal.shape[0]),
        "signal_quality": "partial",
        "method": "scipy-fallback",
    }


def compute_ecg_intervals(signal: np.ndarray, sample_rate: float = 500.0) -> dict[str, Any]:
    """
    Compute HR, PR, QRS, QT, QTc from 12-lead (or partial) ECG using neurokit2 on Lead II.
    """
    if signal is None or signal.size == 0:
        return {
            "signal_quality": "insufficient",
            "hr_bpm": None,
            "pr_ms": None,
            "qrs_ms": None,
            "qt_ms": None,
            "qtc_ms": None,
            "num_leads": 0,
            "method": "no-signal",
        }

    try:
        import neurokit2 as nk
    except ImportError:
        logger.warning("neurokit2 not installed — scipy HR fallback")
        return _scipy_hr_fallback(signal, sample_rate)

    lead = signal[1] if signal.shape[0] > 1 else signal[0]
    sr = int(round(sample_rate))

    try:
        cleaned = nk.ecg_clean(lead, sampling_rate=sr)
        signals_df, _info = nk.ecg_process(cleaned, sampling_rate=sr)
    except Exception as e:
        logger.warning("neurokit2 ecg_process failed: %s — scipy fallback", e)
        return _scipy_hr_fallback(signal, sample_rate)

    required = ("ECG_Rate", "ECG_Q_Peaks", "ECG_S_Peaks")
    if not all(c in signals_df.columns for c in required):
        logger.warning("neurokit2 output missing columns %s — partial fallback", required)
        return _scipy_hr_fallback(signal, sample_rate)

    hr_bpm = float(signals_df["ECG_Rate"].dropna().mean())

    # PR: prefer P onset → Q; fallback P peak → Q peak
    if _COL_PR_START in signals_df.columns and _COL_PR_END in signals_df.columns:
        pr_ms = _interval_ms(signals_df, _COL_PR_START, _COL_PR_END, sample_rate)
    else:
        pr_ms = _interval_ms(signals_df, _FALLBACK_PR[0], _FALLBACK_PR[1], sample_rate)

    qrs_ms = _interval_ms(signals_df, _COL_QRS_START, _COL_QRS_END, sample_rate)
    qt_ms = None
    if _COL_QT_END in signals_df.columns:
        qt_ms = _interval_ms(signals_df, _COL_QT_START, _COL_QT_END, sample_rate)

    qtc_ms = _bazett(qt_ms, hr_bpm)

    return {
        "hr_bpm": round(hr_bpm, 1),
        "pr_ms": round(pr_ms, 1) if pr_ms is not None else None,
        "qrs_ms": round(qrs_ms, 1) if qrs_ms is not None else None,
        "qt_ms": round(qt_ms, 1) if qt_ms is not None else None,
        "qtc_ms": round(qtc_ms, 1) if qtc_ms is not None else None,
        "num_leads": int(signal.shape[0]),
        "signal_quality": "good",
        "method": "neurokit2",
    }
