"""Tests for TorchXRayVision ensemble and chest pipeline wiring."""

from __future__ import annotations

import unittest

import pytest


class TestTxrvConstants(unittest.TestCase):
    def test_rsna_not_in_ensemble(self):
        from txrv_utils import FALLBACK_SECONDARY, PRIMARY_WEIGHTS, SECONDARY_WEIGHTS

        for s in (PRIMARY_WEIGHTS, SECONDARY_WEIGHTS, FALLBACK_SECONDARY):
            self.assertNotIn("rsna", s.lower())


@pytest.mark.parametrize(
    "weights",
    [
        "densenet121-res224-all",
        "densenet121-res224-chex",
    ],
)
def test_valid_labels_have_txrv_label_map_entry(weights: str):
    """Every trained slot (truthy pathology name) maps for correlation/report."""
    import torchxrayvision as xrv

    from txrv_utils import TXRV_LABEL_MAP, _load_txrv

    model = _load_txrv(weights)
    labs = model.pathologies
    for i, name in enumerate(xrv.datasets.default_pathologies):
        if i < len(labs) and labs[i]:
            assert name in TXRV_LABEL_MAP, f"missing TXRV_LABEL_MAP for {name!r} ({weights})"


def test_txrv_tensor_shape(sample_xray_path: str, sample_rgb_xray_path: str):
    import torch

    from txrv_utils import txrv_tensor_from_filepath

    for fp in (sample_xray_path, sample_rgb_xray_path):
        t = txrv_tensor_from_filepath(fp)
        assert t.shape == (1, 1, 224, 224)
        assert t.dtype == torch.float32
        assert t.min() >= -1100 and t.max() <= 1100


def test_cxr_smoke_ensemble(sample_xray_path: str):
    from txrv_utils import ensemble_txrv

    scores, agreement, models_used = ensemble_txrv(sample_xray_path)
    assert isinstance(scores, dict)
    assert len(scores) >= 0
    assert 0.0 <= agreement <= 1.0
    assert len(models_used) == 2
    assert "TorchXRayVision-DenseNet121-all" in models_used[0]


class TestChestPipelineReturn(unittest.TestCase):
    def test_run_chest_pipeline_schema_safe_keys(self):
        import pipeline_chest as pc

        # Avoid loading heavy weights twice: patch ensemble
        fake_scores = {"pleural_effusion": 0.2, "no_finding": 0.9}
        fake_models = [
            "TorchXRayVision-DenseNet121-all",
            "TorchXRayVision-DenseNet121-chex",
        ]

        def fake_ensemble(_fp: str):
            return fake_scores, 0.8, fake_models

        real = pc.ensemble_txrv
        pc.ensemble_txrv = fake_ensemble  # type: ignore
        try:
            from unittest.mock import patch

            with patch("pipeline_chest.generate_heatmap", return_value="/heatmaps/x.png"):
                out = pc.run_chest_pipeline("/tmp/nonexistent.png", "job-smoke")
        finally:
            pc.ensemble_txrv = real

        allowed_extra = {
            "modality",
            "detected_region",
            "findings",
            "impression",
            "pathology_scores",
            "structures",
            "confidence",
            "heatmap_url",
            "models_used",
            "disclaimer",
        }
        self.assertTrue(set(out.keys()) <= allowed_extra)
        self.assertNotIn("ensemble_agreement", out)
        self.assertNotIn("narrative", out)
        self.assertIsInstance(out["confidence"], str)
