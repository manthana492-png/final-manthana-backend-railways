from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path


class TestXrayKimiPolicy(unittest.TestCase):
    def test_attach_narrative_requires_kimi(self):
        backend_root = Path(__file__).resolve().parents[1]
        inference_path = backend_root / "services" / "01_body_xray" / "inference.py"
        spec = importlib.util.spec_from_file_location("xray_inference", inference_path)
        assert spec and spec.loader
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)

        original = inference.XRAY_REQUIRE_KIMI_NARRATIVE
        original_fn = inference._optional_llm_narrative
        try:
            inference.XRAY_REQUIRE_KIMI_NARRATIVE = True
            inference._optional_llm_narrative = lambda **_: ""
            with self.assertRaises(RuntimeError):
                inference.attach_narrative(
                    {
                        "structures": {},
                        "pathology_scores": {},
                        "impression": "test",
                        "models_used": [],
                    }
                )
        finally:
            inference.XRAY_REQUIRE_KIMI_NARRATIVE = original
            inference._optional_llm_narrative = original_fn

