from __future__ import annotations

import totalseg_idle as ti


def test_idle_policy_snapshot_keys():
    snap = ti.idle_policy_snapshot()
    assert "totalseg_idle_empty_cache_sec" in snap
    assert "reaper_enabled" in snap


def test_empty_cache_check_disabled_when_sec_zero(monkeypatch):
    monkeypatch.setattr(ti, "IDLE_EMPTY_CACHE_SEC", 0.0)
    out = ti.run_totalseg_idle_empty_cache_check()
    assert out.get("enabled") is False


def test_touch_updates_activity(monkeypatch):
    monkeypatch.setattr(ti, "IDLE_EMPTY_CACHE_SEC", 999999.0)
    ti.touch_totalseg_gpu_activity()
    out = ti.run_totalseg_idle_empty_cache_check()
    assert out.get("emptied") is False
    assert out.get("enabled") is True
