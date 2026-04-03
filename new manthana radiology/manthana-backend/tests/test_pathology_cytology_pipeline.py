"""Tests for pathology and cytology pipelines (mocked; CPU-safe)."""
from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PATHOLOGY_ROOT = ROOT / "services" / "05_pathology"
CYTOLOGY_ROOT = ROOT / "services" / "11_cytology"
SHARED_ROOT = ROOT / "shared"


def _pop_inference_modules() -> None:
    for m in ("config", "inference", "preprocessing"):
        sys.modules.pop(m, None)


def _tile_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (64, 64), (210, 180, 140)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestPathologyFindingsList:
    def test_build_findings_returns_list(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(PATHOLOGY_ROOT))
        from inference import _build_findings_from_dsmil

        result = _build_findings_from_dsmil(
            {
                "malignancy_score": 0.75,
                "benign_score": 0.10,
                "inflammation_score": 0.10,
                "necrosis_score": 0.05,
            }
        )
        assert isinstance(result, list)
        assert all("label" in f for f in result)
        assert any(f.get("severity") == "critical" for f in result)

    def test_safe_scores_coerces_non_numeric(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(PATHOLOGY_ROOT))
        from inference import _safe_scores

        out = _safe_scores({"malignancy_score": 0.8, "analysis_pending": True, "tissue_type": "x"})
        assert all(isinstance(v, float) for v in out.values())


class TestPathologyBridge:
    def test_bridge_function_exists(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(PATHOLOGY_ROOT))
        import inference

        assert hasattr(inference, "run_pathology_pipeline_b64")

    def test_bridge_calls_run_pipeline(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(PATHOLOGY_ROOT))
        import inference as path_inference

        with patch.object(path_inference, "run_pipeline") as mock_rp:
            mock_rp.return_value = {
                "modality": "pathology",
                "findings": [{"label": "Malignant", "severity": "critical", "confidence": 84.0}],
                "impression": "High-grade carcinoma",
                "pathology_scores": {"malignancy_score": 0.84},
                "structures": {"tissue_source": "breast"},
                "models_used": ["Virchow (Apache 2.0)", "DSMIL-MIL"],
                "disclaimer": "AI second opinion only.",
                "confidence": "medium",
            }
            from inference import run_pathology_pipeline_b64

            result = run_pathology_pipeline_b64(
                image_b64=_tile_b64(),
                patient_context={"tissue_source": "breast"},
            )
        assert mock_rp.called
        assert isinstance(result["findings"], list)


class TestCytologyAnalyzeCells:
    def test_pap_smear_has_bethesda(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(CYTOLOGY_ROOT))
        import inference as cyto_inference
        import numpy as np

        with patch.object(cyto_inference, "dsmil_slide_scores") as mock_dsmil:
            mock_dsmil.return_value = {
                "malignancy_score": 0.05,
                "benign_score": 0.90,
                "inflammation_score": 0.03,
                "necrosis_score": 0.02,
                "classification_status": "dsmil_mil",
            }
            from inference import _analyze_cells

            embeddings = [np.random.randn(2560).astype("float32") for _ in range(25)]
            result = _analyze_cells(embeddings, "pap_smear")
        assert "bethesda_category" in result
        assert result["specimen_type"] == "pap_smear"
        assert "adequacy" in result

    def test_fnac_no_bethesda(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(CYTOLOGY_ROOT))
        import inference as cyto_inference
        import numpy as np

        with patch.object(cyto_inference, "dsmil_slide_scores") as mock_dsmil:
            mock_dsmil.return_value = {
                "malignancy_score": 0.10,
                "benign_score": 0.85,
                "inflammation_score": 0.05,
                "necrosis_score": 0.0,
                "classification_status": "dsmil_mil",
            }
            from inference import _analyze_cells

            embeddings = [np.random.randn(2560).astype("float32") for _ in range(10)]
            result = _analyze_cells(embeddings, "fnac")
        assert "bethesda_category" not in result

    def test_high_malignancy_critical_pap(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(CYTOLOGY_ROOT))
        import inference as cyto_inference
        import numpy as np

        with patch.object(cyto_inference, "dsmil_slide_scores") as mock_dsmil:
            mock_dsmil.return_value = {
                "malignancy_score": 0.72,
                "benign_score": 0.10,
                "inflammation_score": 0.10,
                "necrosis_score": 0.08,
                "classification_status": "dsmil_mil",
            }
            from inference import _analyze_cells

            embeddings = [np.random.randn(2560).astype("float32") for _ in range(30)]
            result = _analyze_cells(embeddings, "pap_smear")
        assert result.get("is_critical") is True

    def test_low_cellularity_unsatisfactory(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(CYTOLOGY_ROOT))
        import inference as cyto_inference
        import numpy as np

        with patch.object(cyto_inference, "dsmil_slide_scores") as mock_dsmil:
            mock_dsmil.return_value = {
                "malignancy_score": 0.05,
                "benign_score": 0.90,
                "inflammation_score": 0.03,
                "necrosis_score": 0.02,
                "classification_status": "dsmil_mil",
            }
            from inference import _analyze_cells

            embeddings = [np.random.randn(2560).astype("float32") for _ in range(3)]
            result = _analyze_cells(embeddings, "pap_smear")
        ad = result["adequacy"].lower()
        assert "unsatisfactory" in ad or "insufficient" in ad


class TestCytologyBridge:
    def test_bridge_calls_run_pipeline(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(CYTOLOGY_ROOT))
        import inference as cyto_inference

        with patch.object(cyto_inference, "run_pipeline") as mock_rp:
            mock_rp.return_value = {
                "modality": "cytology",
                "findings": [{"label": "NILM", "severity": "clear", "confidence": 88.0}],
                "impression": "Negative for intraepithelial lesion",
                "pathology_scores": {
                    "NILM": 0.88,
                    "malignant": 0.02,
                    "tb_suggestive": 0.0,
                    "is_critical": 0.0,
                },
                "structures": {"specimen_type": "pap_smear", "bethesda_category": "NILM"},
                "models_used": ["Virchow (Apache 2.0)", "DSMIL-MIL"],
                "disclaimer": "AI second opinion only.",
                "confidence": "medium",
            }
            from inference import run_cytology_pipeline_b64

            result = run_cytology_pipeline_b64(
                image_b64=_tile_b64(),
                patient_context={"specimen_type": "pap_smear"},
            )
        assert mock_rp.called
        assert isinstance(result["findings"], list)
        assert isinstance(result["structures"], dict)


class TestZeroClawPathologyCytology:
    def test_tools_registered(self) -> None:
        sys.path.insert(0, str(SHARED_ROOT))
        from zeroclaw_tools import TOOLS

        names = [t["function"]["name"] for t in TOOLS]
        assert "analyze_pathology" in names
        assert "analyze_cytology" in names

    def test_pathology_executor_never_raises(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(PATHOLOGY_ROOT))
        import inference as path_inference

        sys.path.insert(0, str(SHARED_ROOT))
        with patch.object(path_inference, "run_pathology_pipeline_b64") as mock_run:
            mock_run.return_value = {"modality": "pathology", "findings": []}
            from zeroclaw_tools import _exec_analyze_pathology

            out = _exec_analyze_pathology(
                image_b64=_tile_b64(),
                patient_context_json=json.dumps({"tissue_source": "breast"}),
            )
        assert isinstance(out, dict)
        assert out.get("modality") == "pathology"

    def test_cytology_executor_never_raises(self) -> None:
        _pop_inference_modules()
        sys.path.insert(0, str(CYTOLOGY_ROOT))
        import inference as cyto_inference

        sys.path.insert(0, str(SHARED_ROOT))
        with patch.object(cyto_inference, "run_cytology_pipeline_b64") as mock_run:
            mock_run.return_value = {"modality": "cytology", "findings": []}
            from zeroclaw_tools import _exec_analyze_cytology

            out = _exec_analyze_cytology(
                image_b64=_tile_b64(),
                patient_context_json=json.dumps({"specimen_type": "pap_smear"}),
            )
        assert isinstance(out, dict)
        assert out.get("modality") == "cytology"
