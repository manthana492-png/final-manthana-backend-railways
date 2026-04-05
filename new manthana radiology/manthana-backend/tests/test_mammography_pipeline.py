"""Unit tests for mammography inference (mocked Mirai — no HF download)."""

from __future__ import annotations

import base64
import importlib.util
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MAMMO_ROOT = ROOT / "services" / "12_mammography"
SHARED_ROOT = ROOT / "shared"


def _setup_mammo_path() -> None:
    for m in ("config", "inference"):
        sys.modules.pop(m, None)
    if str(MAMMO_ROOT) not in sys.path:
        sys.path.insert(0, str(MAMMO_ROOT))
    if str(SHARED_ROOT) not in sys.path:
        sys.path.insert(0, str(SHARED_ROOT))


def _jpeg_b64(size=(256, 256)) -> str:
    buf = BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mammo_zeroclaw_module():
    """Same module object zeroclaw_tools loads for mammography (for patch.object)."""
    from zeroclaw_tools import _MAMMOGRAPHY, _MAMMO_INF_MODULE

    inf_path = os.path.join(_MAMMOGRAPHY, "inference.py")
    if _MAMMO_INF_MODULE in sys.modules:
        return sys.modules[_MAMMO_INF_MODULE]
    spec = importlib.util.spec_from_file_location(_MAMMO_INF_MODULE, inf_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MAMMO_INF_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


def _tmp_jpeg() -> str:
    buf = BytesIO()
    Image.new("RGB", (256, 256), (200, 200, 200)).save(buf, format="JPEG")
    f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    f.write(buf.getvalue())
    f.close()
    return f.name


class TestRiskCategory:
    def setup_method(self) -> None:
        _setup_mammo_path()

    def test_low(self) -> None:
        from inference import _risk_category

        assert _risk_category(0.01) == "low"

    def test_average(self) -> None:
        from inference import _risk_category

        assert _risk_category(0.02) == "average"

    def test_moderate(self) -> None:
        from inference import _risk_category

        assert _risk_category(0.04) == "moderate"

    def test_high(self) -> None:
        from inference import _risk_category

        assert _risk_category(0.06) == "high"


class TestBuildFindings:
    def setup_method(self) -> None:
        _setup_mammo_path()

    def test_single_image_returns_warning_no_risk_scores(self) -> None:
        from inference import _build_findings

        result = _build_findings(
            {"available": False, "reason": "Four views required"},
            has_four_views=False,
        )
        assert isinstance(result, list)
        assert result[0]["severity"] == "warning"
        desc = (result[0].get("description") or "").lower()
        assert "four" in desc or "view" in desc

    def test_high_risk_is_critical(self) -> None:
        from inference import _build_findings

        result = _build_findings(
            {
                "available": True,
                "cancer_risk_1yr": 0.02,
                "cancer_risk_2yr": 0.03,
                "cancer_risk_3yr": 0.04,
                "cancer_risk_5yr": 0.07,
                "risk_category": "high",
                "is_high_risk": 1.0,
                "views_used": ["L-CC", "L-MLO", "R-CC", "R-MLO"],
            },
            has_four_views=True,
        )
        assert result[0]["severity"] == "critical"
        assert "HIGH" in result[0]["label"].upper()

    def test_low_risk_is_clear(self) -> None:
        from inference import _build_findings

        result = _build_findings(
            {
                "available": True,
                "cancer_risk_1yr": 0.005,
                "cancer_risk_2yr": 0.008,
                "cancer_risk_3yr": 0.01,
                "cancer_risk_5yr": 0.012,
                "risk_category": "low",
                "is_high_risk": 0.0,
                "views_used": ["L-CC", "L-MLO", "R-CC", "R-MLO"],
            },
            has_four_views=True,
        )
        assert result[0]["severity"] == "clear"

    def test_india_note_always_present_four_views(self) -> None:
        from inference import _build_findings

        result = _build_findings(
            {
                "available": True,
                "cancer_risk_1yr": 0.01,
                "cancer_risk_2yr": 0.015,
                "cancer_risk_3yr": 0.02,
                "cancer_risk_5yr": 0.025,
                "risk_category": "average",
                "is_high_risk": 0.0,
                "views_used": ["L-CC", "L-MLO", "R-CC", "R-MLO"],
            },
            has_four_views=True,
        )
        blob = " ".join(
            (f.get("description", "") + " " + f.get("label", "")) for f in result
        ).lower()
        assert "india" in blob or "indian" in blob or "population" in blob

    def test_severity_never_normal(self) -> None:
        from inference import _build_findings

        cases = [
            ({"available": False, "reason": "x"}, False),
            (
                {
                    "available": True,
                    "cancer_risk_1yr": 0.01,
                    "cancer_risk_2yr": 0.015,
                    "cancer_risk_3yr": 0.02,
                    "cancer_risk_5yr": 0.025,
                    "risk_category": "average",
                    "is_high_risk": 0.0,
                    "views_used": ["L-CC", "L-MLO", "R-CC", "R-MLO"],
                },
                True,
            ),
        ]
        for scores, four in cases:
            findings = _build_findings(scores, has_four_views=four)
            for f in findings:
                assert f.get("severity") != "normal"


class TestSafeScores:
    def setup_method(self) -> None:
        _setup_mammo_path()

    def test_non_numeric_coerced(self) -> None:
        from inference import _safe_scores

        result = _safe_scores(
            {
                "cancer_risk_5yr": 0.05,
                "risk_category": "high",
                "is_high_risk": True,
                "views_used": ["L-CC"],
            }
        )
        assert all(isinstance(v, float) for v in result.values())


class TestPipelineMocked:
    def setup_method(self) -> None:
        _setup_mammo_path()

    @patch("inference._mammo_narrative_openrouter", return_value=("Test report.", []))
    @patch("inference._run_mirai")
    def test_four_views_calls_mirai(self, mock_mirai, _mock_claude) -> None:
        mock_mirai.return_value = {
            "available": True,
            "cancer_risk_1yr": 0.01,
            "cancer_risk_2yr": 0.015,
            "cancer_risk_3yr": 0.02,
            "cancer_risk_5yr": 0.025,
            "risk_category": "average",
            "is_high_risk": 0.0,
            "views_used": ["L-CC", "L-MLO", "R-CC", "R-MLO"],
        }
        from inference import run_pipeline

        tmp = _tmp_jpeg()
        try:
            result = run_pipeline(
                filepath=tmp,
                patient_context={
                    "views": {
                        "L-CC": tmp,
                        "L-MLO": tmp,
                        "R-CC": tmp,
                        "R-MLO": tmp,
                    }
                },
            )
            assert mock_mirai.called
            assert result["structures"]["has_four_views"] is True
            assert isinstance(result["findings"], list)
            assert all(isinstance(v, float) for v in result["pathology_scores"].values())
        finally:
            os.unlink(tmp)

    @patch("inference._mammo_narrative_openrouter", return_value=("Visual only.", []))
    @patch("inference._run_mirai")
    def test_single_image_skips_mirai(self, mock_mirai, _mock_claude) -> None:
        from inference import run_pipeline

        tmp = _tmp_jpeg()
        try:
            result = run_pipeline(filepath=tmp, patient_context={})
            assert not mock_mirai.called
            assert result["structures"]["has_four_views"] is False
            assert "birads_4_or_above" in result["pathology_scores"]
        finally:
            os.unlink(tmp)


class TestSingleImageNoFabricatedScores:
    def setup_method(self) -> None:
        _setup_mammo_path()

    @patch("inference._mammo_narrative_openrouter", return_value=("", []))
    @patch("inference._run_mirai")
    def test_single_image_pathology_scores_include_heuristics(self, mock_mirai, _mock_narr) -> None:
        from inference import run_pipeline

        tmp = _tmp_jpeg()
        try:
            r = run_pipeline(filepath=tmp, patient_context={})
            assert not mock_mirai.called
            ps = r["pathology_scores"]
            assert "birads_4_or_above" in ps
            assert "malignancy_confidence" in ps
        finally:
            os.unlink(tmp)


class TestZeroClawMammography:
    def setup_method(self) -> None:
        sys.modules.pop("zeroclaw_tools", None)
        if str(SHARED_ROOT) not in sys.path:
            sys.path.insert(0, str(SHARED_ROOT))

    def test_tool_registered(self) -> None:
        from zeroclaw_tools import TOOLS

        names = [t["function"]["name"] for t in TOOLS]
        assert "analyze_mammography" in names

    def test_executor_returns_dict(self) -> None:
        _setup_mammo_path()
        mammo_mod = _mammo_zeroclaw_module()

        with patch.object(
            mammo_mod,
            "run_mammography_pipeline_b64",
            return_value={
                "modality": "mammography",
                "findings": [],
                "pathology_scores": {},
                "structures": {"has_four_views": False},
            },
        ):
            from zeroclaw_tools import _exec_analyze_mammography

            result = _exec_analyze_mammography(
                image_b64=_jpeg_b64(),
                patient_context_json='{"age": 50, "birads_density": 2}',
            )
            assert isinstance(result, dict)
            assert result.get("modality") == "mammography"

    def test_executor_graceful_on_inference_error(self) -> None:
        _setup_mammo_path()
        mammo_mod = _mammo_zeroclaw_module()

        with patch.object(
            mammo_mod,
            "run_mammography_pipeline_b64",
            side_effect=RuntimeError("simulated failure"),
        ):
            from zeroclaw_tools import _exec_analyze_mammography

            result = _exec_analyze_mammography(
                image_b64=_jpeg_b64(),
                patient_context_json="{}",
            )
            assert isinstance(result, dict)
            assert result.get("available") is False
            assert "message" in result or "reason" in result
