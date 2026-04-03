import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))


class TestComp2compRunner(unittest.TestCase):
    def test_aaq_returns_diameter_keys(self):
        from comp2comp_runner import run_aaq

        vol = np.random.randint(-200, 400, size=(32, 32, 32)).astype(np.float32)
        r = run_aaq(volume=vol, affine=None)
        self.assertIn("max_aorta_diameter_mm", r)
        self.assertIn("aaa_detected", r)
        self.assertIsInstance(r["max_aorta_diameter_mm"], (int, float))

    def test_bmd_returns_scores(self):
        from comp2comp_runner import run_bmd

        vol = np.random.randint(50, 400, size=(16, 16, 16)).astype(np.float32)
        r = run_bmd(volume=vol)
        self.assertIn("bmd_score", r)
        self.assertIn("low_bmd_flag", r)
