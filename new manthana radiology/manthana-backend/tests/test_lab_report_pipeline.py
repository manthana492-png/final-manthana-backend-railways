"""Unit and lightweight integration tests for lab report pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import analyzer as lab_analyzer
from critical_values import check_critical_values, normalize_labs_for_critical
import inference as lab_inference


class TestParseClinicalNotes(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(lab_analyzer.parse_clinical_notes(""), {})
        self.assertEqual(lab_analyzer.parse_clinical_notes("   "), {})

    def test_parses_keys(self):
        raw = "age:45; gender: male; FASTING: no; medications: metformin"
        out = lab_analyzer.parse_clinical_notes(raw)
        self.assertEqual(out.get("age"), "45")
        self.assertEqual(out.get("gender"), "male")
        self.assertEqual(out.get("fasting"), "no")
        self.assertEqual(out.get("medications"), "metformin")


class TestCriticalValues(unittest.TestCase):
    def test_ignores_unconfirmed_units(self):
        labs = normalize_labs_for_critical(
            {"glucose": {"value": 600, "unit": "mg/dl", "unit_confirmed": False}}
        )
        self.assertEqual(check_critical_values(labs), [])

    def test_glucose_critical_high_mg_dl(self):
        labs = normalize_labs_for_critical(
            {"glucose": {"value": 600, "unit": "mg/dl", "unit_confirmed": True}}
        )
        alerts = check_critical_values(labs)
        self.assertTrue(any("CRITICAL HIGH" in a for a in alerts))

    def test_glucose_ok_when_in_range(self):
        labs = normalize_labs_for_critical(
            {"glucose": {"value": 100, "unit": "mg/dl", "unit_confirmed": True}}
        )
        self.assertEqual(check_critical_values(labs), [])


def test_extract_test_results_from_sample_line() -> None:
    text = "Haemoglobin (Hb) 9.2 g/dL 13.0–17.0 L\nESR (Westergren) 88 mm/hr 0–15 (M) HH"
    rows = lab_inference.extract_test_results_from_text(text)
    assert len(rows) >= 2
    assert any("Haemoglobin" in r["test_name"] for r in rows)


def test_run_lab_bad_b64_returns_available_false() -> None:
    out = lab_inference.run_lab_report_pipeline_b64("@@@not-valid-base64!!!")
    assert out.get("available") is False


def test_zeroclaw_lab_bad_b64_dict_style_call() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
    from zeroclaw_tools import TOOL_EXECUTORS

    r = TOOL_EXECUTORS["analyze_lab_report"](
        {
            "document_b64": "not_valid_base64!!",
            "patient_context_json": "{}",
        }
    )
    assert r.get("available") is False
    assert r.get("reason") == "bad_b64"


def test_analysis_response_lab_shape() -> None:
    shared = Path(__file__).resolve().parents[1] / "shared"
    sys.path.insert(0, str(shared))
    from schemas import AnalysisResponse

    payload = {
        "job_id": "j-lab-1",
        "modality": "lab_report",
        "findings": [{"label": "x", "severity": "info", "confidence": 50.0}],
        "pathology_scores": {"tb_pattern_confidence": 0.1},
        "structures": {
            "report_type": "LFT",
            "test_results": [],
            "critical_values": [],
            "abnormal_count": 0,
            "critical_count": 0,
            "patterns_detected": [],
            "india_context": {},
            "narrative_report": "",
            "ocr_confidence": None,
            "page_count": 1,
        },
        "is_critical": False,
    }
    AnalysisResponse(**payload)


@patch.object(lab_analyzer, "PARROTV_AVAILABLE", False)
@patch.object(lab_analyzer, "_interpret_raw_text")
def test_analyze_lab_report_txt_mocked_interp(mock_interp, tmp_path):
    mock_interp.return_value = {
        "findings": [],
        "impression": "Synthetic test impression.",
        "pathology_scores": {},
        "structures": [],
        "detected_region": "lab_report",
    }
    p = tmp_path / "labs.txt"
    p.write_text("CBC panel placeholder text for testing.", encoding="utf-8")
    out = lab_analyzer.analyze_lab_report(str(p))
    assert out.get("modality") == "lab_report"
    assert "parser_used" in out
    mock_interp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
