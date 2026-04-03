import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "report_assembly"))


class TestCorrelationEngine(unittest.TestCase):
    def test_heart_failure_pattern(self):
        from correlation_engine import find_correlations

        results = [
            {
                "modality": "xray",
                "result": {"pathology_scores": {"pleural_effusion": 0.9}},
            },
            {
                "modality": "lab_report",
                "result": {"pathology_scores": {"BNP": 500.0}},
            },
        ]
        out = find_correlations(results)
        names = [o["pattern"] for o in out]
        self.assertIn("Heart Failure Indicators", names)

    def test_empty_results(self):
        from correlation_engine import find_correlations

        self.assertEqual(find_correlations([]), [])

    def test_mammography_high_risk_and_pathology(self):
        from correlation_engine import find_correlations

        results = [
            {
                "modality": "mammography",
                "result": {
                    "pathology_scores": {
                        "is_high_risk": 1.0,
                        "cancer_risk_5yr": 0.07,
                    },
                    "structures": {
                        "has_four_views": True,
                        "risk_category": "high",
                    },
                },
            },
            {
                "modality": "pathology",
                "result": {"pathology_scores": {"malignancy_score": 0.8}},
            },
        ]
        out = find_correlations(results)
        names = [o["pattern"] for o in out]
        self.assertIn("Mammography High Risk — Specialist Review", names)
        self.assertIn("Mammography + Pathology Concordant Malignancy", names)

    def test_ecg_long_qt_rule(self):
        from correlation_engine import find_correlations

        results = [
            {
                "modality": "ecg",
                "result": {
                    "pathology_scores": {"qtc_ms": 520.0},
                    "structures": {"intervals": {"qtc_ms": 520.0}},
                },
            },
        ]
        out = find_correlations(results)
        names = [o["pattern"] for o in out]
        self.assertIn("Long QT — Drug / Electrolyte Check", names)

    def test_ecg_afib_young_rhd_rule(self):
        from correlation_engine import find_correlations

        results = [
            {
                "modality": "ecg",
                "result": {
                    "pathology_scores": {
                        "afib_confidence": 0.85,
                        "patient_age": 32.0,
                    },
                },
            },
        ]
        out = find_correlations(results)
        names = [o["pattern"] for o in out]
        self.assertIn("AFib Young Patient — RHD Screening", names)
