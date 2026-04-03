from __future__ import annotations

import time
import unittest


class _DummyParam:
    def __init__(self, is_cuda: bool):
        self.is_cuda = is_cuda


class _DummyModel:
    def __init__(self, is_cuda: bool):
        self._is_cuda = is_cuda

    def parameters(self):
        yield _DummyParam(self._is_cuda)

    def cpu(self):
        self._is_cuda = False
        return self


class TestTxrvIdlePolicy(unittest.TestCase):
    def test_gpu_only_moves_residency_to_cpu(self):
        import txrv_utils as tu

        old_idle = tu.IDLE_UNLOAD_SEC
        old_mode = tu.UNLOAD_MODE
        try:
            tu.IDLE_UNLOAD_SEC = 1.0
            tu.UNLOAD_MODE = "gpu_only"
            tu._models["densenet121-res224-all"] = _DummyModel(is_cuda=True)
            tu._model_residency["densenet121-res224-all"] = "gpu"
            tu._last_used_epoch_s["densenet121-res224-all"] = time.time() - 10.0
            out = tu.run_idle_unload_check()
            self.assertIn("densenet121-res224-all", out["unloaded"])
            self.assertEqual(tu._model_residency["densenet121-res224-all"], "cpu")
            self.assertIn("densenet121-res224-all", tu._models)
        finally:
            tu._models.clear()
            tu._model_residency.clear()
            tu._last_used_epoch_s.clear()
            tu.IDLE_UNLOAD_SEC = old_idle
            tu.UNLOAD_MODE = old_mode

    def test_full_mode_drops_model_from_memory(self):
        import txrv_utils as tu

        old_idle = tu.IDLE_UNLOAD_SEC
        old_mode = tu.UNLOAD_MODE
        try:
            tu.IDLE_UNLOAD_SEC = 1.0
            tu.UNLOAD_MODE = "full"
            tu._models["densenet121-res224-all"] = _DummyModel(is_cuda=True)
            tu._model_residency["densenet121-res224-all"] = "gpu"
            tu._last_used_epoch_s["densenet121-res224-all"] = time.time() - 10.0
            out = tu.run_idle_unload_check()
            self.assertIn("densenet121-res224-all", out["unloaded"])
            self.assertNotIn("densenet121-res224-all", tu._models)
        finally:
            tu._models.clear()
            tu._model_residency.clear()
            tu._last_used_epoch_s.clear()
            tu.IDLE_UNLOAD_SEC = old_idle
            tu.UNLOAD_MODE = old_mode

