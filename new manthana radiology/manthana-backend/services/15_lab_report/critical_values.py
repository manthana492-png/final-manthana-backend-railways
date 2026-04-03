"""
Deterministic critical-value screening for lab reports.
Only evaluates analytes with unit_confirmed=True (parser-explicit units).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# (canonical_key, low, high, unit_substrings, action)
# Use separate rows per unit band to avoid duplicate dict keys for glucose/creatinine.
CRITICAL_RULES: List[Tuple[str, Optional[float], Optional[float], Tuple[str, ...], str]] = [
    ("hemoglobin", 5.0, None, ("g/dl",), "Emergency transfusion threshold — correlate clinically"),
    ("hemoglobin", 50.0, None, ("g/l",), "Critical anaemia pattern — verify unit (g/L)"),
    ("platelets", 20.0, None, ("10^3", "10³", "/ul", "/µl", "k/ul"), "Spontaneous bleeding risk"),
    ("wbc", None, 30.0, ("10^3", "10³", "/ul", "/µl", "k/ul"), "Severe leukocytosis — clinical correlation"),
    ("potassium", 2.5, 6.5, ("meq/l", "mmol/l"), "Cardiac arrhythmia risk"),
    ("sodium", 120.0, 160.0, ("meq/l", "mmol/l"), "Neurologic emergency risk"),
    ("glucose", 2.5, 27.8, ("mmol/l",), "Hypoglycaemia / severe hyperglycaemia — urgent correlation"),
    ("glucose", 45.0, 500.0, ("mg/dl",), "Hypoglycaemia / severe hyperglycaemia — urgent correlation"),
    ("creatinine", None, 10.0, ("mg/dl",), "Severe renal impairment — urgent correlation"),
    ("creatinine", None, 884.0, ("µmol/l", "umol/l"), "Severe renal impairment — urgent correlation"),
    ("troponin_i", None, 0.4, ("ng/ml", "ng/l"), "Possible myocardial injury — correlate ECG"),
    ("troponin_t", None, 0.1, ("ng/ml",), "Possible myocardial injury — correlate ECG"),
    ("bnp", None, 400.0, ("pg/ml",), "Heart failure pattern — correlate imaging"),
    ("inr", None, 5.0, ("inr", "ratio"), "Major bleeding risk — correlate anticoagulation"),
]


def _norm_unit(u: str) -> str:
    return (u or "").lower().replace(" ", "").replace("μ", "µ")


def _canonical_key(name: str) -> str:
    k = name.lower().strip().replace(" ", "_").replace("-", "_")
    aliases = {
        "hgb": "hemoglobin",
        "hb": "hemoglobin",
        "plt": "platelets",
        "platelet": "platelets",
        "potassium": "potassium",
        "k": "potassium",
        "sodium": "sodium",
        "na": "sodium",
        "glucose": "glucose",
        "fbs": "glucose",
        "rbs": "glucose",
        "creatinine": "creatinine",
        "creat": "creatinine",
        "trop_i": "troponin_i",
        "trop_t": "troponin_t",
        "bnp": "bnp",
        "inr": "inr",
    }
    return aliases.get(k, k)


def normalize_labs_for_critical(labs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a dict of analyte -> {value, unit, unit_confirmed} for critical checks.
    Accepts flat numbers, dicts with value/unit, or nested parser output.
    """
    out: Dict[str, Any] = {}
    if not labs or not isinstance(labs, dict):
        return out

    def add_entry(name: str, value: Any, unit: Optional[str], confirmed: bool) -> None:
        ck = _canonical_key(name)
        if ck not in out:
            out[ck] = {"value": value, "unit": unit or "", "unit_confirmed": confirmed}

    for key, val in labs.items():
        if isinstance(val, dict):
            v = val.get("value", val.get("result"))
            u = val.get("unit") or val.get("units") or ""
            if "unit_confirmed" in val:
                uc = bool(val.get("unit_confirmed"))
            elif u:
                uc = _unit_explicit(str(u))
            else:
                uc = False
            if v is not None:
                try:
                    fv = float(v)
                    add_entry(str(key), fv, str(u), uc)
                except (TypeError, ValueError):
                    continue
        else:
            try:
                fv = float(val)
                add_entry(str(key), fv, "", False)
            except (TypeError, ValueError):
                continue

    return out


def _unit_explicit(unit_str: str) -> bool:
    u = _norm_unit(unit_str)
    if not u:
        return False
    return bool(re.search(r"[a-z%µ/]", u))


def check_critical_values(labs_normalized: Dict[str, Any]) -> List[str]:
    """Return human-readable alert strings for critical values."""
    alerts: List[str] = []
    for key, entry in labs_normalized.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("unit_confirmed"):
            continue
        val = entry.get("value")
        if not isinstance(val, (int, float)):
            continue
        unit = _norm_unit(str(entry.get("unit") or ""))
        ck = _canonical_key(key)

        for rule_key, low, high, unit_kws, action in CRITICAL_RULES:
            if _canonical_key(rule_key) != ck:
                continue
            if not unit_kws:
                match = True
            else:
                match = any(kw and kw in unit for kw in unit_kws)
            if ck == "inr" and not unit:
                match = True
            if not match:
                continue
            if low is not None and val < low:
                alerts.append(
                    f"CRITICAL LOW {key}: {val} {entry.get('unit', '')} — {action}"
                )
            if high is not None and val > high:
                alerts.append(
                    f"CRITICAL HIGH {key}: {val} {entry.get('unit', '')} — {action}"
                )
    return alerts
