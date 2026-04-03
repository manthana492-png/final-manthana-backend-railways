import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))


class TestMemoryManager(unittest.TestCase):
    def test_eviction_order(self):
        from model_loader import ManagedModel, ModelMemoryManager

        mm = ModelMemoryManager(gpu_budget_gb=25.0)
        mm.loaded.clear()

        a = MagicMock(spec=ManagedModel)
        a.cache_name = "a"
        a.vram_gb = 10.0
        a.last_used = 1.0
        a.is_loaded.return_value = True
        a.is_on_gpu.return_value = True
        a.offload_to_cpu = MagicMock()

        b = MagicMock(spec=ManagedModel)
        b.cache_name = "b"
        b.vram_gb = 10.0
        b.last_used = 2.0
        b.is_loaded.return_value = True
        b.is_on_gpu.return_value = True
        b.offload_to_cpu = MagicMock()

        c = MagicMock(spec=ManagedModel)
        c.cache_name = "c"
        c.vram_gb = 10.0
        c.last_used = 3.0
        c.is_loaded.return_value = True
        c.is_on_gpu.return_value = True
        c.offload_to_cpu = MagicMock()

        mm.loaded = {"a": a, "b": b}
        mm._evict_coldest(exclude_cache_name="c")
        a.offload_to_cpu.assert_called_once()
