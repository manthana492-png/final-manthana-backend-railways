"""Correlation rules added for lab_report cross-modality E2E."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPORT_ASM = BACKEND_ROOT / "services" / "report_assembly"
SHARED = BACKEND_ROOT / "shared"


@pytest.fixture(autouse=True)
def _paths() -> None:
    for p in (SHARED, REPORT_ASM):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def test_tb_lab_plus_spine_rule() -> None:
    from correlation_engine import find_correlations

    r = find_correlations(
        [
            {"modality": "lab_report", "result": {"pathology_scores": {"tb_pattern_confidence": 0.82}}},
            {"modality": "spine_neuro", "result": {"pathology_scores": {"pott_disease_confidence": 0.71}}},
        ]
    )
    names = [c.get("name", "") for c in r]
    assert any("Lab TB Pattern + Spine" in n for n in names), names


def test_ckd_dm_single_modality_rule() -> None:
    from correlation_engine import find_correlations

    r = find_correlations(
        [
            {
                "modality": "lab_report",
                "result": {
                    "pathology_scores": {
                        "renal_impairment_score": 0.78,
                        "diabetes_control_score": 0.72,
                    }
                },
            },
        ]
    )
    names = [c.get("name", "") for c in r]
    assert any("Diabetic Nephropathy" in n for n in names), names


def test_hepatic_plus_liver_lesion_rule() -> None:
    from correlation_engine import find_correlations

    r = find_correlations(
        [
            {"modality": "lab_report", "result": {"pathology_scores": {"hepatic_injury_score": 0.91}}},
            {"modality": "abdominal_ct", "result": {"pathology_scores": {"liver_lesion_confidence": 0.68}}},
        ]
    )
    names = [c.get("name", "") for c in r]
    assert any("HBsAg Reactive + CT Liver" in n or "HCC" in n for n in names), names


def test_tb_lab_plus_cxr_rule() -> None:
    from correlation_engine import find_correlations

    r = find_correlations(
        [
            {"modality": "lab_report", "result": {"pathology_scores": {"tb_pattern_confidence": 0.7}}},
            {"modality": "cxr", "result": {"pathology_scores": {"infiltrate_confidence": 0.55}}},
        ]
    )
    names = [c.get("name", "") for c in r]
    assert any("Lab TB Pattern + CXR" in n for n in names), names


def test_anaemia_cardiomegaly_rule() -> None:
    from correlation_engine import find_correlations

    r = find_correlations(
        [
            {"modality": "lab_report", "result": {"pathology_scores": {"anaemia_severity_score": 0.75}}},
            {"modality": "xray", "result": {"pathology_scores": {"cardiomegaly": 0.5}}},
        ]
    )
    names = [c.get("name", "") for c in r]
    assert any("Severe Anaemia + CXR" in n for n in names), names
