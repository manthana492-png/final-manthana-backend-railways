import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))


class TestTriage(unittest.TestCase):
    def test_heuristic_triage_returns_keys(self):
        from triage import _triage_xray_heuristic

        fd, path = tempfile.mkstemp(suffix=".png")
        import os

        os.close(fd)
        try:
            from PIL import Image

            Image.fromarray((np.random.rand(64, 64) * 255).astype("uint8")).save(path)
            r = _triage_xray_heuristic(path)
            self.assertIn("needs_deep", r)
            self.assertIn("triage_time_ms", r)
        finally:
            os.unlink(path)
