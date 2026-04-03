import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))


class TestTotalsegRunner(unittest.TestCase):
    def test_structure_list_from_result(self):
        from totalseg_runner import structure_list_from_result

        r = {
            "structure_names": ["liver", "spleen"],
            "mask_paths": {},
            "output_dir": ".",
        }
        self.assertEqual(structure_list_from_result(r), ["liver", "spleen"])
