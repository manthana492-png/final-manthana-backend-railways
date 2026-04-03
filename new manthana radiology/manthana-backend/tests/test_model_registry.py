import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))


class TestModelRegistry(unittest.TestCase):
    def test_rollback_swaps(self):
        from model_registry import ModelRegistry

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            data = {
                "txrv_primary": {
                    "current": {"model_id": "A"},
                    "previous": {"model_id": "B"},
                }
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            with patch("model_registry.REGISTRY_PATH", path):
                r = ModelRegistry()
                r.rollback("txrv_primary")
                self.assertEqual(r.entries["txrv_primary"]["current"]["model_id"], "B")
        finally:
            os.unlink(path)
