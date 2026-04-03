"""MRI pipeline remediation: TotalSeg MR stems, spine task mapping, schema, PACS MSK routing."""

from __future__ import annotations

import importlib.util
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "services" / "pacs_bridge"))


def _load_shared_schemas():
    """Avoid collision with gateway/schemas when the full test suite runs."""
    path = ROOT / "shared" / "schemas.py"
    spec = importlib.util.spec_from_file_location("manthana_shared_schemas_mri", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMriRemediation(unittest.TestCase):
    def test_total_mr_stem_count(self):
        from totalseg_label_maps import TOTALSEG_TOTAL_MR_VOLUME_KEYS

        self.assertEqual(len(TOTALSEG_TOTAL_MR_VOLUME_KEYS), 50)

    def test_map_organ_key_total_mr(self):
        from totalseg_label_maps import map_organ_key

        self.assertEqual(map_organ_key("brain", "total_mr"), "brain_cm3")
        self.assertEqual(map_organ_key("liver", "total_mr"), "liver_cm3")
        self.assertIsNone(map_organ_key("not_a_stem", "total_mr"))

    def test_map_organ_key_vertebrae_mr(self):
        from totalseg_label_maps import map_organ_key

        self.assertEqual(map_organ_key("vertebrae_L3", "vertebrae_mr"), "vertebrae_L3_cm3")
        self.assertEqual(map_organ_key("sacrum", "vertebrae_mr"), "sacrum_cm3")
        self.assertIsNone(map_organ_key("liver", "vertebrae_mr"))

    def test_analysis_response_brain_mri_findings(self):
        mod = _load_shared_schemas()
        AnalysisResponse = mod.AnalysisResponse
        Finding = mod.Finding

        r = AnalysisResponse(
            job_id="j1",
            modality="brain_mri",
            findings=[
                Finding(label="Test", severity="info", confidence=90.0, description="x"),
            ],
        )
        self.assertEqual(r.findings[0].label, "Test")

    def test_pacs_msk_mr_routes_unsupported(self):
        from dicom_router import BODY_PART_MR

        self.assertEqual(BODY_PART_MR["KNEE"], "unsupported_mr_msk")
        self.assertEqual(BODY_PART_MR["ANKLE"], "unsupported_mr_msk")

    def test_brain_pipeline_findings_roundtrip_analysis_response(self):
        """Real `_build_brain_findings` output must validate as AnalysisResponse.findings (Phase C)."""
        sys.path.insert(0, str(ROOT / "services" / "02_brain_mri"))
        sys.path.insert(0, str(ROOT / "shared"))
        sys.modules.pop("inference", None)
        from inference import _build_brain_findings

        mod = _load_shared_schemas()
        findings = _build_brain_findings(
            tot_ok=False,
            tot_names=[],
            synth={"available": False, "reason": "unit_test"},
            prima={"available": False, "reason": "prima_not_configured"},
            degraded_2d=False,
            filepath="/tmp/unitbrain.nii.gz",
            clinical_notes="",
        )
        self.assertIsInstance(findings, list)
        self.assertGreater(len(findings), 0)
        payload = {
            "job_id": "u1",
            "modality": "brain_mri",
            "findings": [
                f.model_dump() if hasattr(f, "model_dump") else f for f in findings
            ],
            "impression": "Unit test impression.",
        }
        mod.AnalysisResponse.model_validate(payload)

    def test_spine_detect_is_mri_alias_and_nifti_default(self):
        sys.path.insert(0, str(ROOT / "services" / "10_spine_neuro"))
        sys.path.insert(0, str(ROOT / "shared"))
        sys.modules.pop("inference", None)
        from inference import _detect_is_mri, detect_is_mri

        self.assertIs(_detect_is_mri, detect_is_mri)
        self.assertFalse(detect_is_mri("/nonexistent_brain.nii.gz", None))

    def test_spine_empty_segmentation_finding_warning(self):
        sys.path.insert(0, str(ROOT / "services" / "10_spine_neuro"))
        sys.path.insert(0, str(ROOT / "shared"))
        sys.modules.pop("inference", None)
        from inference import _build_spine_findings

        out = _build_spine_findings(
            [],
            [24.0],
            degraded=False,
            volumes_cm3={},
            is_mri=True,
            tot_task="vertebrae_mr",
        )
        self.assertTrue(any("unavailable" in f.label.lower() for f in out))

    def test_prima_mlp_failure_returns_empty_scores(self):
        import numpy as np

        sys.path.insert(0, str(ROOT / "shared"))
        import prima_mlp_head as pm

        with unittest.mock.patch.object(pm.np, "asarray", side_effect=RuntimeError("force_failure")):
            scores = pm.run_prima_mlp(np.zeros(128, dtype=np.float32))
        self.assertEqual(scores, {})


if __name__ == "__main__":
    unittest.main()
