"""
tests/test_usg_pipeline.py
USG pipeline unit + ZeroClaw integration tests
Run: pytest tests/test_usg_pipeline.py -v
"""

import sys
import os
import base64
import io
import json

import numpy as np
from PIL import Image

_USG_SERVICE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "services", "09_ultrasound"
)
_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")
sys.path.insert(0, os.path.abspath(_USG_SERVICE_DIR))
sys.path.insert(0, os.path.abspath(_SHARED_DIR))


def _make_test_b64(width: int = 224, height: int = 224) -> str:
    """Create a synthetic grayscale USG-like image encoded as base64 JPEG."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[50:150, 50:170] = 180
    arr[160:190, 80:130] = 30
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_bad_b64() -> str:
    return "not_valid_base64!!!"


class TestUSGPipelineB64:
    def test_valid_image_returns_response(self):
        from inference import run_usg_pipeline_b64

        result = run_usg_pipeline_b64(_make_test_b64())
        assert result.get("modality") == "ultrasound"
        assert "findings" in result
        assert "pathology_scores" in result

    def test_bad_b64_returns_available_false(self):
        from inference import run_usg_pipeline_b64

        result = run_usg_pipeline_b64(_make_bad_b64())
        assert result["available"] is False
        assert result["reason"] == "bad_b64"

    def test_empty_b64_returns_available_false(self):
        from inference import run_usg_pipeline_b64

        result = run_usg_pipeline_b64("")
        assert result["available"] is False

    def test_patient_context_json_accepted(self):
        from inference import run_usg_pipeline_b64

        ctx = json.dumps({"age": 45, "sex": "M", "complaint": "RUQ pain"})
        result = run_usg_pipeline_b64(_make_test_b64(), patient_context_json=ctx)
        assert result.get("modality") == "ultrasound"

    def test_pathology_scores_keys_present(self):
        from inference import run_usg_pipeline_b64, enrich_usg_pipeline_output

        result = run_usg_pipeline_b64(_make_test_b64())
        result = enrich_usg_pipeline_output(result)
        ps = result["pathology_scores"]
        for key in (
            "ascites_indicator",
            "free_fluid_present",
            "liver_echogenicity_high",
            "renal_echogenicity_high",
            "parenchymal_heterogeneity",
            "image_quality_adequate",
        ):
            assert key in ps, f"Missing pathology_score key: {key}"

    def test_free_fluid_indicator_non_negative(self):
        from inference import run_usg_pipeline_b64

        result = run_usg_pipeline_b64(_make_test_b64())
        score = result["pathology_scores"].get("free_fluid_indicator", -1)
        assert 0.0 <= score <= 1.0

    def test_models_used_obfuscated(self):
        from inference import run_usg_pipeline_b64

        result = run_usg_pipeline_b64(_make_test_b64())
        for m in result.get("models_used", []):
            assert "openus" not in m.lower()
            assert "echocare" not in m.lower()


class TestUSGZeroClaw:
    def test_zeroclaw_tool_registered(self):
        from zeroclaw_tools import TOOLS

        names = [t["function"]["name"] for t in TOOLS]
        assert "analyze_usg" in names

    def test_zeroclaw_exec_bad_b64_dict(self):
        from zeroclaw_tools import _exec_analyze_usg

        result = _exec_analyze_usg({"image_b64": "bad!!"})
        assert result["available"] is False
        assert result["reason"] in ("bad_b64", "bad_input")

    def test_zeroclaw_exec_string_input(self):
        from zeroclaw_tools import _exec_analyze_usg

        payload = json.dumps({"image_b64": _make_test_b64()})
        result = _exec_analyze_usg(payload)
        assert result.get("modality") == "ultrasound"


class TestUSGCorrelationRules:
    def _make_usg_response(
        self, free_fluid: float = 0.0, liver_high: bool = False, lymph: bool = False
    ) -> dict:
        findings = "Moderate free fluid in peritoneal cavity. "
        if lymph:
            findings += "Mesenteric lymphadenopathy noted."
        if liver_high:
            findings += " Liver echogenicity increased."
        return {
            "modality": "ultrasound",
            "pathology_scores": {
                "free_fluid_present": free_fluid > 0.3,
                "free_fluid_indicator": free_fluid,
                "ascites_indicator": free_fluid,
                "liver_echogenicity_high": liver_high,
                "parenchymal_heterogeneity": 0.3,
                "image_quality_adequate": True,
            },
            "findings": findings,
        }

    def _make_lab_response(
        self,
        alt_high: bool = False,
        albumin_low: bool = False,
        creatinine_high: bool = False,
    ) -> dict:
        return {
            "modality": "lab_report",
            "pathology_scores": {
                "alt_elevated": alt_high,
                "albumin_low": albumin_low,
                "creatinine_elevated": creatinine_high,
            },
            "structures": {},
        }

    def test_tb_peritonitis_rule_fires(self):
        from correlation_engine import find_correlations

        usg = self._make_usg_response(free_fluid=0.6, lymph=True)
        results = find_correlations([{"modality": "ultrasound", "result": usg}])
        patterns = [r["name"] for r in results]
        assert any("USG_FREE_FLUID_LYMPH_TB" in p for p in patterns)

    def test_decompensated_liver_rule_fires(self):
        from correlation_engine import find_correlations

        usg = self._make_usg_response(free_fluid=0.7)
        lab = self._make_lab_response(albumin_low=True)
        results = find_correlations(
            [
                {"modality": "ultrasound", "result": usg},
                {"modality": "lab_report", "result": lab},
            ]
        )
        patterns = [r["name"] for r in results]
        assert any("USG_ASCITES_LAB_ALBUMIN" in p for p in patterns)

    def test_hepatic_parenchymal_rule_fires(self):
        from correlation_engine import find_correlations

        usg = self._make_usg_response(liver_high=True)
        lab = self._make_lab_response(alt_high=True)
        results = find_correlations(
            [
                {"modality": "ultrasound", "result": usg},
                {"modality": "lab_report", "result": lab},
            ]
        )
        patterns = [r["name"] for r in results]
        assert any("USG_LIVER_HIGH_ECHO_LAB_ALT" in p for p in patterns)

    def test_no_false_fire_on_normal_usg(self):
        from correlation_engine import find_correlations

        usg = self._make_usg_response(free_fluid=0.0, liver_high=False)
        results = find_correlations([{"modality": "ultrasound", "result": usg}])
        usg_patterns = [r["name"] for r in results if r["name"].startswith("USG_")]
        assert len(usg_patterns) == 0

