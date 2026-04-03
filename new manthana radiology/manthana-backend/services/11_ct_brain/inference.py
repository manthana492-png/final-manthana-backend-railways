"""Manthana — CT Brain (NCCT): deploy-time TorchScript, CI dummy, or explicit weights_required mode."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_backend_root = Path(__file__).resolve().parents[2]
for _shared in (Path("/app/shared"), _backend_root / "shared"):
    if _shared.is_dir():
        sys.path.insert(0, str(_shared))
        break

from ct_brain_gpu_idle import (
    ensure_ct_brain_idle_reaper_started,
    idle_policy_snapshot,
    touch_ct_brain_gpu_activity,
)
from disclaimer import DISCLAIMER
from preprocessing.ct_loader import is_degraded_single_slice, load_ct_volume
from schemas import Finding

logger = logging.getLogger("manthana.ct_brain")
PIPELINE_VERSION = "manthana-ct-brain-v1"

_ts_model: Any = None
_ts_load_ms: float | None = None
_ci_dummy_module: Any = None


def _narrative_policy() -> str:
    return os.environ.get("CT_BRAIN_NARRATIVE_POLICY", "kimi_then_anthropic").strip().lower()


def _critical_threshold() -> float:
    return float(os.environ.get("CT_BRAIN_CRITICAL_THRESHOLD", "0.5") or "0.5")


def _torch_device():
    import torch

    if os.getenv("CT_BRAIN_DEVICE", "").strip().lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _window_brain(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol, dtype=np.float32)
    center, width = 40.0, 80.0
    lo = center - width / 2
    hi = center + width / 2
    w = np.clip(v, lo, hi)
    return (w - lo) / max(hi - lo, 1e-6)


def _volume_to_tensor5d(windowed: np.ndarray, target_d: int = 48, target_hw: int = 256):
    """(Z,Y,X) -> (1,1,D,H,W) float32."""
    import torch
    import torch.nn.functional as F

    v = np.asarray(windowed, dtype=np.float32)
    if v.ndim == 2:
        v = v[np.newaxis, ...]
    elif v.ndim != 3:
        raise ValueError(f"expected 2D/3D volume, got shape {v.shape}")
    t = torch.from_numpy(v)[None, None, ...]
    t = F.interpolate(t, size=(target_d, target_hw, target_hw), mode="trilinear", align_corners=False)
    return t


def _load_torchscript() -> Any | None:
    global _ts_model, _ts_load_ms
    path = (os.environ.get("CT_BRAIN_TORCHSCRIPT_PATH") or "").strip()
    if not path or not os.path.isfile(path):
        return None
    import torch

    t0 = time.perf_counter()
    dev = _torch_device()
    _ts_model = torch.jit.load(path, map_location=dev)
    _ts_model.eval()
    _ts_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info("Loaded CT Brain TorchScript in %.2f ms on %s", _ts_load_ms, dev)
    return _ts_model


def _run_ci_dummy(t: Any) -> float:
    import torch
    import torch.nn as nn

    global _ci_dummy_module
    if _ci_dummy_module is None:

        class _CiDummy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AdaptiveAvgPool3d(1)
                self.fc = nn.Linear(1, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.pool(x).view(x.size(0), -1)
                return self.fc(y)

        _ci_dummy_module = _CiDummy().to(_torch_device())
    m = _ci_dummy_module
    t = t.to(next(m.parameters()).device)
    with torch.no_grad():
        o = m(t)
        prob = torch.sigmoid(o.view(-1)[0]).item()
    return float(prob)


def _run_torchscript(t: Any) -> float | None:
    import torch

    m = _ts_model or _load_torchscript()
    if m is None:
        return None
    dev = _torch_device()
    t = t.to(dev)
    with torch.no_grad():
        out = m(t)
    if isinstance(out, (list, tuple)):
        out = out[0]
    out = out.float().reshape(-1)
    if out.numel() == 1:
        return float(torch.sigmoid(out[0]).item())
    if out.numel() >= 2:
        return float(torch.softmax(out, dim=-1)[1].item())
    return None


def _anthropic_narrative(system: str, user_text: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return ""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        msg = client.messages.create(
            model=model,
            max_tokens=1200,
            system=system[:20000],
            messages=[{"role": "user", "content": user_text[:120000]}],
        )
        block = msg.content[0]
        return getattr(block, "text", str(block)).strip()
    except ImportError:
        logger.warning("anthropic not installed; skip CT brain narrative")
        return ""
    except Exception as e:
        logger.warning("CT brain Anthropic narrative failed: %s", e)
        return ""


def _call_ct_brain_narrative(
    *,
    impression: str,
    findings: list,
    pathology_scores: dict,
    patient_context: dict | None,
) -> tuple[str, list[str]]:
    policy = _narrative_policy()
    tags: list[str] = []
    if policy in ("off", "none", "disabled", "0"):
        return "", tags

    system = (
        "You are a neuroradiologist assistant. Write a concise NCCT brain report-style narrative "
        "using ONLY the structured JSON. Flag emergency patterns if scores/findings support them; "
        "do not invent hemorrhage if ich_probability is absent or inference_mode is weights_required."
    )
    user_text = (
        f"IMPRESSION (pipeline):\n{impression}\n\n"
        f"FINDINGS:\n{json.dumps(findings, indent=2)[:8000]}\n\n"
        f"PATHOLOGY_SCORES:\n{json.dumps(pathology_scores, indent=2)[:8000]}\n\n"
        f"PATIENT_CONTEXT:\n{json.dumps(patient_context or {}, indent=2)[:4000]}\n\n"
        f"CT_BRAIN_CLINICAL:\n{json.dumps((patient_context or {}).get('ct_brain_clinical_context') or {}, indent=2)[:2000]}"
    )

    key = (os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or "").strip()
    if key and policy in ("kimi_then_anthropic", "kimi_only"):
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed; skip CT brain Kimi narrative")
        else:
            base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1").strip()
            model = (
                os.environ.get("KIMI_CT_MODEL", "").strip()
                or os.environ.get("KIMI_MODEL", "kimi-k2.5").strip()
            )
            try:
                client = OpenAI(api_key=key, base_url=base_url)
                r = client.chat.completions.create(
                    model=model,
                    max_tokens=1200,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_text},
                    ],
                )
                out = (r.choices[0].message.content or "").strip()
                if out:
                    tags.append("Kimi-narrative-CT-Brain")
                    return out, tags
            except Exception as e:
                logger.warning("CT brain Kimi narrative failed: %s", e)

    if policy == "kimi_only":
        return "", tags

    ant = _anthropic_narrative(system, user_text)
    if ant:
        tags.append("Anthropic-narrative-CT-Brain")
        return ant, tags
    return "", tags


def is_loaded() -> dict:
    path = (os.environ.get("CT_BRAIN_TORCHSCRIPT_PATH") or "").strip()
    ci = os.getenv("CT_BRAIN_CI_DUMMY_MODEL", "").lower() in ("1", "true", "yes")
    return {
        "torchscript_configured": bool(path),
        "torchscript_file_present": bool(path and os.path.isfile(path)),
        "ci_dummy_enabled": ci,
        "pipeline_version": PIPELINE_VERSION,
        **idle_policy_snapshot(),
    }


def run_pipeline(
    filepath: str,
    job_id: str,
    series_dir: str = "",
    source_modality: str = "",
    patient_context: dict | None = None,
) -> dict:
    ensure_ct_brain_idle_reaper_started()
    t0 = time.perf_counter()
    volume, meta, series_avail = load_ct_volume(filepath, series_dir=series_dir or None)
    degraded = is_degraded_single_slice(volume)
    meta = meta if isinstance(meta, dict) else {}
    meta_mod = str(meta.get("modality") or "").upper()

    models_used: list[str] = []
    pathology_scores: dict[str, Any] = {"series_available": bool(series_avail)}
    inference_mode = "weights_required"
    ich_prob: float | None = None
    tensor_in = None

    try:
        w = _window_brain(volume)
        tensor_in = _volume_to_tensor5d(w)
    except Exception as e:
        logger.warning("CT brain preprocess failed: %s", e)

    if os.getenv("CT_BRAIN_CI_DUMMY_MODEL", "").lower() in ("1", "true", "yes"):
        if tensor_in is not None:
            ich_prob = _run_ci_dummy(tensor_in)
        else:
            ich_prob = 0.0
        inference_mode = "ci_dummy"
        models_used.append("CT-Brain-CI-Dummy")
    elif (os.environ.get("CT_BRAIN_TORCHSCRIPT_PATH") or "").strip():
        if tensor_in is not None:
            ich_prob = _run_torchscript(tensor_in)
        if ich_prob is not None:
            inference_mode = "torchscript"
            models_used.append("CT-Brain-TorchScript")
        else:
            inference_mode = "weights_required"
            models_used.append("CT-Brain-TorchScript-MissingOrFailed")
    else:
        models_used.append("CT-Brain-NoWeights")

    pathology_scores["inference_mode"] = inference_mode
    if ich_prob is not None:
        pathology_scores["ich_probability"] = round(float(ich_prob), 4)

    findings: list[Finding] = []
    if degraded:
        findings.append(
            Finding(
                label="Degraded input — limited volumetric analysis",
                description="Thin or single-slice input. NCCT brain algorithms are most reliable on a full axial series.",
                severity="warning",
                confidence=100.0,
                region="Brain",
            )
        )

    if meta_mod == "MR":
        findings.append(
            Finding(
                label="MRI / non-CT DICOM hint",
                description=(
                    "DICOM header suggests non-CT modality. CT Brain pipeline is validated for NCCT; "
                    "use Brain MRI workflow for MR studies."
                ),
                severity="warning",
                confidence=95.0,
                region="Brain",
            )
        )

    if source_modality and str(source_modality).upper() not in ("", "CT"):
        findings.append(
            Finding(
                label="Source modality hint",
                description=f"source_modality={source_modality!r} — confirm this is an NCCT head study.",
                severity="info",
                confidence=90.0,
                region="Brain",
            )
        )

    if inference_mode == "weights_required":
        findings.append(
            Finding(
                label="CT Brain model not configured",
                description=(
                    "Set CT_BRAIN_TORCHSCRIPT_PATH to a validated TorchScript bundle at deploy time. "
                    "No ich_probability is emitted in this mode."
                ),
                severity="warning",
                confidence=100.0,
                region="Brain",
            )
        )

    thr = _critical_threshold()
    if ich_prob is not None and ich_prob >= thr:
        findings.append(
            Finding(
                label="Suspected intracranial hemorrhage (model flag)",
                description=(
                    f"Model ich_probability={ich_prob:.3f} (threshold {thr}). "
                    "Emergency clinical correlation and neuroradiology review required."
                ),
                severity="critical",
                confidence=min(99.0, 50.0 + ich_prob * 50.0),
                region="Brain",
            )
        )

    impression = "NCCT brain analysis complete. Clinical correlation required."
    if inference_mode == "weights_required":
        impression = "NCCT brain received — AI hemorrhage model not configured; no automated ICH score."
    elif inference_mode == "ci_dummy":
        impression = "NCCT brain CI dummy inference — not for clinical use."

    findings_out = [f.model_dump() if isinstance(f, Finding) else f for f in findings]
    narrative, narr_tags = _call_ct_brain_narrative(
        impression=impression,
        findings=findings_out,
        pathology_scores=pathology_scores,
        patient_context=patient_context,
    )
    models_used.extend(narr_tags)

    structures: dict[str, Any] = {
        "narrative_report": narrative or "",
        "narrative_policy": _narrative_policy(),
        "inference_mode": inference_mode,
        "input_degraded": degraded,
        "dicom_modality_header": meta_mod or None,
        "algorithm_version": {
            "pipeline_version": PIPELINE_VERSION,
            "torchscript_load_ms": _ts_load_ms,
        },
        "inference_wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "idle_policy": idle_policy_snapshot(),
    }

    if narrative and len(narrative) > 40 and inference_mode != "weights_required":
        impression = narrative[:320].strip() + ("…" if len(narrative) > 320 else "")

    touch_ct_brain_gpu_activity()

    return {
        "modality": "ct_brain",
        "findings": findings_out,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": "medium",
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }
