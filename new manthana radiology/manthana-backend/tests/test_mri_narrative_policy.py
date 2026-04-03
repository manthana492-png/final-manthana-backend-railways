"""MRI narrative policy env (no live LLM calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def brain_mri_inference():
    for p in (ROOT / "services" / "02_brain_mri", ROOT / "shared"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    sys.modules.pop("inference", None)
    import inference as inf

    return inf


def test_mri_narrative_policy_off_returns_empty(brain_mri_inference, monkeypatch):
    monkeypatch.setenv("MRI_NARRATIVE_POLICY", "off")
    from pathlib import Path

    inf = brain_mri_inference
    text, tags = inf._call_mri_narrative(
        pathology_scores={"brain_cm3": 1200.0},
        patient_context={},
        image_b64=None,
        prompt_path=Path(inf.__file__).resolve().parent / "prompts" / "brain_mri_system.md",
        impression="Test",
        findings=[],
    )
    assert text == ""
    assert tags == []


def test_mri_narrative_anthropic_only_skips_kimi_without_key(brain_mri_inference, monkeypatch):
    monkeypatch.setenv("MRI_NARRATIVE_POLICY", "anthropic_only")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from pathlib import Path

    inf = brain_mri_inference
    text, tags = inf._call_mri_narrative(
        pathology_scores={},
        patient_context={},
        image_b64="fakeb64wouldfailanyway",
        prompt_path=Path(inf.__file__).resolve().parent / "prompts" / "brain_mri_system.md",
        impression="x",
        findings=[],
    )
    assert text == ""
    assert tags == []
