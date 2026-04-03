"""Integration-style tests for report assembly with CXR pathology keys."""

from __future__ import annotations

import unittest


class TestReportCorrelationIntegration(unittest.TestCase):
    def test_find_correlations_with_canonical_xray_keys(self):
        from correlation_engine import find_correlations

        results = [
            {
                "modality": "xray",
                "result": {
                    "pathology_scores": {
                        "pleural_effusion": 0.85,
                        "cardiomegaly": 0.4,
                    },
                },
            },
            {
                "modality": "lab_report",
                "result": {"pathology_scores": {"BNP": 400.0}},
            },
        ]
        out = find_correlations(results)
        self.assertIsInstance(out, list)
