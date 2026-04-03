"""Spine/neuro and mammography pipeline + correlation smoke tests."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPINE_ROOT = ROOT / "services" / "10_spine_neuro"
MAMMO_ROOT = ROOT / "services" / "12_mammography"
SHARED_ROOT = ROOT / "shared"
REPORT_ASM = ROOT / "services" / "report_assembly"


def _b64_png(condition: str = "spine") -> str:
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    if condition == "spine":
        arr[100:400, 200:300] = [80, 50, 130]
        arr[150:250, 300:360] = [200, 200, 200]
    else:
        arr[:, :] = [30, 30, 30]
        arr[300:500, 300:500] = [180, 180, 180]
        arr[350:450, 350:450] = [230, 230, 230]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestSpinePipeline:
    def test_spine_import(self) -> None:
        for m in ("config", "inference"):
            sys.modules.pop(m, None)
        sys.path.insert(0, str(SPINE_ROOT))
        sys.path.insert(0, str(SHARED_ROOT))
        import inference as spine_inf

        fns = [f for f in dir(spine_inf) if "pipeline" in f and "b64" in f]
        assert fns, f"No pipeline b64 function: {dir(spine_inf)}"

    def test_spine_pipeline_runs(self) -> None:
        for m in ("config", "inference"):
            sys.modules.pop(m, None)
        sys.path.insert(0, str(SPINE_ROOT))
        sys.path.insert(0, str(SHARED_ROOT))
        import inference as si

        fn = getattr(si, "run_spine_neuro_pipeline_b64", None) or getattr(si, "run_pipeline_b64", None)
        if fn is None:
            pytest.skip("no b64 pipeline")
        result = fn(
            image_b64=_b64_png("spine"),
            patient_context_json=json.dumps({"age": 40, "sex": "M", "clinical_history": "back pain"}),
        )
        assert isinstance(result, dict)
        if result.get("available") is False:
            pytest.skip(result.get("message", "unavailable"))
        assert "findings" in result
        assert "pathology_scores" in result
        assert "structures" in result
        st = result["structures"]
        assert "vertebral_levels_assessed" in st
        assert "narrative_report" in st

    def test_spine_severity_values(self) -> None:
        for m in ("config", "inference"):
            sys.modules.pop(m, None)
        sys.path.insert(0, str(SPINE_ROOT))
        sys.path.insert(0, str(SHARED_ROOT))
        import inference as si

        fn = getattr(si, "run_spine_neuro_pipeline_b64", None) or getattr(si, "run_pipeline_b64", None)
        if fn is None:
            pytest.skip()
        result = fn(image_b64=_b64_png("spine"), patient_context_json="{}")
        if result.get("available") is False:
            pytest.skip(result.get("message", ""))
        valid = {"critical", "warning", "info", "clear"}
        for f in result["findings"]:
            sev = f.get("severity") if isinstance(f, dict) else getattr(f, "severity", None)
            assert sev in valid, f"bad severity {sev}"


class TestMammoPipeline:
    def test_mammo_import(self) -> None:
        for m in ("config", "inference"):
            sys.modules.pop(m, None)
        sys.path.insert(0, str(MAMMO_ROOT))
        sys.path.insert(0, str(SHARED_ROOT))
        import inference as mi

        assert any("pipeline" in f and "b64" in f for f in dir(mi))

    @patch("inference._mammo_narrative_kimi_then_claude", return_value=("", []))
    @patch("inference._run_mirai", return_value={"available": False})
    def test_mammo_inference_runs(self, *_mocks: object) -> None:
        for m in ("config", "inference"):
            sys.modules.pop(m, None)
        sys.path.insert(0, str(MAMMO_ROOT))
        sys.path.insert(0, str(SHARED_ROOT))
        import inference as mi

        result = mi.run_mammography_pipeline_b64(
            image_b64=_b64_png("mammo"),
            patient_context_json=json.dumps(
                {"age": 52, "sex": "F", "clinical_history": "palpable lump"}
            ),
        )
        if result.get("available") is False:
            pytest.skip(result.get("message", ""))
        assert "birads_category" in result.get("structures", {})
        assert "malignancy_confidence" in result.get("pathology_scores", {})

    def test_birads4b_is_critical(self) -> None:
        for m in ("config", "inference"):
            sys.modules.pop(m, None)
        sys.path.insert(0, str(MAMMO_ROOT))
        sys.path.insert(0, str(SHARED_ROOT))
        import inference as mi

        fake_st = {
            "view": "MLO",
            "breast_density": "ACR_C",
            "birads_category": "4B",
            "mass_present": True,
            "mass_location": "upper_outer_quadrant",
            "mass_shape": "irregular",
            "mass_margin": "spiculated",
            "calcification_present": False,
            "calcification_morphology": None,
            "calcification_distribution": None,
            "asymmetry_present": False,
            "architectural_distortion": True,
            "axillary_adenopathy": False,
            "is_critical": True,
            "narrative_report": "",
        }
        fake_sc = {
            "malignancy_confidence": 0.85,
            "mass_confidence": 0.82,
            "calcification_confidence": 0.03,
            "birads_4_or_above": 0.9,
            "density_score": 0.7,
        }
        buf = io.BytesIO()
        Image.new("RGB", (256, 256), (200, 200, 200)).save(buf, format="JPEG")
        import tempfile

        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            f.write(buf.getvalue())
            f.close()
            with patch("inference._heuristic_mammo_from_path", return_value=(fake_st, fake_sc)):
                with patch("inference._mammo_narrative_kimi_then_claude", return_value=("", [])):
                    with patch(
                        "inference._run_mirai",
                        return_value={"available": False},
                    ):
                        result = mi.run_pipeline(f.name, patient_context={}, image_b64="")
        finally:
            os.unlink(f.name)
        assert result.get("is_critical") is True


class TestCorrelationNewRules:
    def setup_method(self) -> None:
        if str(REPORT_ASM) not in sys.path:
            sys.path.insert(0, str(REPORT_ASM))
        if str(SHARED_ROOT) not in sys.path:
            sys.path.insert(0, str(SHARED_ROOT))

    def test_potts_rule_fires(self) -> None:
        from correlation_engine import find_correlations

        results = find_correlations(
            [
                {
                    "modality": "spine_neuro",
                    "result": {"pathology_scores": {"pott_disease_confidence": 0.88}},
                }
            ]
        )
        names = [r.get("name", "") for r in results]
        assert any("Pott" in n for n in names), names

    def test_birads_rule_fires(self) -> None:
        from correlation_engine import find_correlations

        results = find_correlations(
            [
                {
                    "modality": "mammography",
                    "result": {
                        "pathology_scores": {
                            "birads_4_or_above": 0.92,
                            "malignancy_confidence": 0.88,
                        }
                    },
                }
            ]
        )
        names = [r.get("name", "") for r in results]
        assert any("BI-RADS" in n for n in names), names

    def test_mammo_cxr_staging_fires(self) -> None:
        from correlation_engine import find_correlations

        results = find_correlations(
            [
                {
                    "modality": "mammography",
                    "result": {
                        "pathology_scores": {
                            "malignancy_confidence": 0.75,
                            "birads_4_or_above": 0.8,
                        }
                    },
                },
                {"modality": "cxr", "result": {"pathology_scores": {"nodule_confidence": 0.65}}},
            ]
        )
        names = [r.get("name", "") for r in results]
        assert any("Breast" in n for n in names), names
