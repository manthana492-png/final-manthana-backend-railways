"""
ECG rhythm scores from raw 12-lead (or partial) signal — V5 fallback / ensemble path.

Used when fairseq_signals + mimic_iv_ecg_finetuned.pt full pipeline is not loaded.
Produces bounded [0,1] scores (not null) for Indian-doctor-facing UI.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("manthana.ecg_rhythm")

# PhysioNet-style label keys aligned with services/13_ecg consumer
RHYTHM_KEYS = (
    "sinus_rhythm",
    "atrial_fibrillation",
    "sinus_tachycardia",
    "sinus_bradycardia",
    "st_elevation",
    "st_depression",
    "lvh",
    "lbbb",
    "rbbb",
)


def rhythm_scores_from_signal(signal: np.ndarray, sample_rate: float = 500.0) -> dict[str, float]:
    """
    Physiology-inspired screening scores from voltage time series.
    Not a substitute for ecg-fm+fine-tuned fairseq inference when available.
    """
    if signal is None or signal.size == 0:
        logger.warning("Empty or null ECG signal — returning zero rhythm scores")
        return {k: 0.0 for k in RHYTHM_KEYS}

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    lead_ii = x[1] if x.shape[0] > 1 else x[0]
    n = lead_ii.size
    # High-pass-ish derivative for QRS / noise
    d = np.diff(lead_ii, prepend=lead_ii[0])

    # HR from peaks
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(
            -lead_ii,
            distance=max(10, int(0.25 * sample_rate)),
            prominence=np.std(lead_ii) * 0.15,
        )
        if len(peaks) > 2:
            rr = np.diff(peaks) / sample_rate
            hr = 60.0 / np.mean(rr) if np.mean(rr) > 1e-6 else 72.0
            rr_cv = float(np.std(rr) / (np.mean(rr) + 1e-6))
        else:
            hr, rr_cv = 72.0, 0.05
    except Exception:
        hr, rr_cv = 72.0, 0.05

    # Bounded scores
    sr = float(np.clip(1.0 - min(rr_cv * 4.0, 1.0), 0.0, 1.0))
    af = float(np.clip(min(rr_cv * 3.5, 1.0), 0.0, 1.0))
    tach = float(np.clip((hr - 95.0) / 40.0, 0.0, 1.0))
    brady = float(np.clip((55.0 - hr) / 30.0, 0.0, 1.0))

    seg = lead_ii[n // 3 : 2 * n // 3]
    base = float(np.median(lead_ii[: n // 5])) if n > 20 else 0.0
    st_proxy = float(np.mean(seg) - base)
    st_e = float(np.clip(st_proxy * 8.0, 0.0, 1.0))
    st_d = float(np.clip(-st_proxy * 8.0, 0.0, 1.0))

    volt = float(np.std(lead_ii))
    lvh = float(np.clip((volt - 0.2) / 0.5, 0.0, 1.0))

    # Wide QRS proxy: energy in derivative
    qrs_w = float(np.mean(np.abs(d)) / (np.std(lead_ii) + 1e-6))
    lbbb = float(np.clip((qrs_w - 1.2) / 2.0, 0.0, 1.0))
    rbbb = float(np.clip((qrs_w - 1.0) / 2.2, 0.0, 1.0))

    return {
        "sinus_rhythm": sr,
        "atrial_fibrillation": af,
        "sinus_tachycardia": tach,
        "sinus_bradycardia": brady,
        "st_elevation": st_e,
        "st_depression": st_d,
        "lvh": lvh,
        "lbbb": lbbb,
        "rbbb": rbbb,
    }


def rhythm_scores_secondary(signal: np.ndarray, sample_rate: float = 500.0) -> dict[str, float]:
    """Second opinion via spectral emphasis — ensembled with primary in V5."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    lead = x[0]
    n = len(lead)
    if n < 64:
        return rhythm_scores_from_signal(signal, sample_rate)

    spec = np.abs(np.fft.rfft(lead))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    af_band = (spec[(freqs >= 3.0) & (freqs <= 8.0)] ** 2).sum()
    total = (spec**2).sum() + 1e-9
    af_boost = float(np.clip(af_band / total * 3.0, 0.0, 1.0))

    base = rhythm_scores_from_signal(signal, sample_rate)
    base["atrial_fibrillation"] = float(
        np.clip(0.5 * base["atrial_fibrillation"] + 0.5 * af_boost, 0.0, 1.0)
    )
    return base


def ensemble_rhythm_dict(a: dict, b: dict) -> dict:
    keys = set(a.keys()) | set(b.keys())
    out = {}
    for k in keys:
        va = float(a.get(k, 0.0) or 0.0)
        vb = float(b.get(k, 0.0) or 0.0)
        out[k] = float(np.clip((va + vb) / 2.0, 0.0, 1.0))
    return out
