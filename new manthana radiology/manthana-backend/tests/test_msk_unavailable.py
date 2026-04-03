"""MSK / abdomen pipelines return normalized production payloads."""

from __future__ import annotations

import unittest
import tempfile
import os

import numpy as np
from PIL import Image


def _temp_png() -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray((np.random.rand(128, 128) * 255).astype("uint8")).save(path)
    return path


class TestMskProductionPipelines(unittest.TestCase):
    def test_bone_pipeline(self):
        import pipeline_bone as pb

        path = _temp_png()
        try:
            out = pb.run_bone_pipeline(path, "j1", "extremity")
        finally:
            os.unlink(path)
        self.assertEqual(out["confidence"], "medium")
        self.assertIsInstance(out["findings"], list)
        self.assertIsInstance(out["pathology_scores"], dict)
        self.assertIsInstance(out["structures"], dict)
        self.assertEqual(out.get("detected_region"), "extremity")

    def test_abdomen_pipeline(self):
        import pipeline_abdomen as pa

        path = _temp_png()
        try:
            out = pa.run_abdomen_pipeline(path, "j2", "abdomen")
        finally:
            os.unlink(path)
        self.assertEqual(out["confidence"], "medium")
        self.assertIsInstance(out["findings"], list)
        self.assertIsInstance(out["pathology_scores"], dict)
        self.assertIsInstance(out["structures"], dict)
        self.assertEqual(out.get("detected_region"), "abdomen")
