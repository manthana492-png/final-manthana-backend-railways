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

try:
    from manthana_paths import backend_root_from_service_file
except ImportError:
    import importlib.util

    _here = Path(__file__).resolve()
    _helpers = []
    if Path("/app/shared/manthana_paths.py").is_file():
        _helpers.append(Path("/app/shared/manthana_paths.py"))
    try:
        _hp = _here.parents[2] / "shared" / "manthana_paths.py"
        if _hp.is_file():
            _helpers.append(_hp)
    except IndexError:
        pass
    backend_root_from_service_file = None  # type: ignore[assignment]
    for _hp in _helpers:
        _spec = importlib.util.spec_from_file_location("_manthana_paths", _hp)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            backend_root_from_service_file = _mod.backend_root_from_service_file  # type: ignore[assignment]
            break
    if backend_root_from_service_file is None:

        def backend_root_from_service_file(p: Path | str) -> Path:  # type: ignore[no-redef]
            x = Path(p).resolve()
            if os.environ.get("MANTHANA_BACKEND_ROOT"):
                return Path(os.environ["MANTHANA_BACKEND_ROOT"])
            if Path("/app/shared").is_dir() and x.parent == Path("/app"):
                return Path("/app")
            return x.parents[2]


_backend_root = backend_root_from_service_file(__file__)
for _shared in (Path("/app/shared"), _backend_root / "shared"):
    if _shared.is_dir():
        sys.path.insert(0, str(_shared))
        break

from ct_brain_gpu_idle import (
    ensure_ct_brain_idle_reaper_started,
    idle_policy_snapshot,
    touch_ct_brain_gpu_activity,
)
from disclaimer import DISCLAIMER, FILM_PHOTO_DISCLAIMER_ADDENDUM
from film_photo_reporting import (
    FILM_PHOTO_NARRATIVE_PREFIX,
    apply_film_photo_pathology_scores,
    attach_film_meta_to_structures,
    cap_confidence_for_film,
    is_film_photo_meta,
    merge_disclaimer_with_film,
)
from hemorrhage_classifier import run_subtype_classifier
from heuristics import run_ct_brain_heuristics
from preprocessing.ct_loader import is_degraded_single_slice, load_ct_volume
from schemas import Finding
from volume_segmentation import run_volume_analysis

import config as ct_brain_config

logger = logging.getLogger("manthana.ct_brain")
PIPELINE_VERSION = "manthana-ct-brain-v1"

_ts_model: Any = None
_ts_load_ms: float | None = None
_ci_dummy_module: Any = None


def _narrative_policy() -> str:
    v = (os.environ.get("CT_BRAIN_NARRATIVE_POLICY", "openrouter") or "openrouter").strip().lower()
    if v in ("off", "none", "disabled", "0"):
        return "off"
    return "openrouter"


def _critical_threshold() -> float:
    return float(os.environ.get("CT_BRAIN_CRITICAL_THRESHOLD", "0.5") or "0.5")


def _derive_confidence(
    inference_mode: str,
    ich_prob: float | None,
    subtype_ok: bool,
    seg_source: str | None,
) -> str:
    if inference_mode == "ci_dummy":
        return "low"
    if inference_mode == "weights_required":
        return "low"
    if ich_prob is None:
        return "medium"
    parts = 1
    if subtype_ok:
        parts += 1
    if seg_source and seg_source not in ("none",):
        parts += 1
    if ich_prob >= _critical_threshold():
        return "high" if parts >= 2 else "medium-high"
    if parts >= 3:
        return "high"
    if parts == 2:
        return "medium-high"
    return "medium"


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


def _call_ct_brain_narrative(
    *,
    impression: str,
    findings: list,
    pathology_scores: dict,
    patient_context: dict | None,
    film_photo: bool = False,
    image_b64_list: list[str] | None = None,
) -> tuple[str, list[str]]:
    if _narrative_policy() == "off":
        return "", []

    system = (
        "You are a neuroradiologist assistant. Write a concise NCCT brain report-style narrative "
        "using ONLY the structured JSON. Flag emergency patterns if scores/findings support them; "
        "do not invent hemorrhage if ich_probability is absent or inference_mode is weights_required."
    )
    if film_photo:
        system = FILM_PHOTO_NARRATIVE_PREFIX + system
    user_text = (
        f"IMPRESSION (pipeline):\n{impression}\n\n"
        f"FINDINGS:\n{json.dumps(findings, indent=2)[:8000]}\n\n"
        f"PATHOLOGY_SCORES:\n{json.dumps(pathology_scores, indent=2)[:8000]}\n\n"
        f"PATIENT_CONTEXT:\n{json.dumps(patient_context or {}, indent=2)[:4000]}\n\n"
        f"CT_BRAIN_CLINICAL:\n{json.dumps((patient_context or {}).get('ct_brain_clinical_context') or {}, indent=2)[:2000]}"
    )

    try:
        from llm_router import llm_router

        # For film-photo mode with multi-image vision
        has_vision_images = film_photo and image_b64_list and len(image_b64_list) > 0
        
        if has_vision_images:
            logger.info("CT brain narrative: using multi-image vision with %d film-photo slices", len(image_b64_list))
            out = llm_router.complete_for_role(
                "narrative_ct",
                system,
                user_text,
                image_b64_list=image_b64_list,
                image_mime="image/png",
                max_tokens=2000,
            )
        else:
            out = llm_router.complete_for_role(
                "narrative_ct",
                system,
                user_text,
                max_tokens=1200,
            )
        txt = (out.get("content") or "").strip()
        if txt:
            return txt, ["OpenRouter-narrative-CT-Brain"]
    except Exception as e:
        logger.warning("CT brain OpenRouter narrative failed: %s", e)
    return "", []


def is_loaded() -> dict:
    path = (os.environ.get("CT_BRAIN_TORCHSCRIPT_PATH") or "").strip()
    sub = (os.environ.get("CT_BRAIN_SUBTYPE_MODEL_PATH") or "").strip()
    seg = (os.environ.get("CT_BRAIN_SEGMENTATION_MODEL_PATH") or "").strip()
    ci = os.getenv("CT_BRAIN_CI_DUMMY_MODEL", "").lower() in ("1", "true", "yes")
    return {
        "torchscript_configured": bool(path),
        "torchscript_file_present": bool(path and os.path.isfile(path)),
        "subtype_model_configured": bool(sub and os.path.isfile(sub)),
        "segmentation_model_configured": bool(seg and os.path.isfile(seg)),
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
    film_photo = is_film_photo_meta(meta)

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

    subtype_scores = None
    if tensor_in is not None:
        subtype_scores = run_subtype_classifier(tensor_in)
    if subtype_scores:
        pathology_scores["hemorrhage_subtypes"] = subtype_scores
        models_used.append("CT-Brain-Subtype-Classifier")

    vol_info = run_volume_analysis(volume, meta, tensor_in)
    pathology_scores["hemorrhage_volume_ml"] = vol_info.get("hemorrhage_volume_ml")
    pathology_scores["ventricle_volume_ml"] = vol_info.get("ventricle_volume_ml")
    pathology_scores["segmentation_source"] = vol_info.get("segmentation_source")
    if vol_info.get("segmentation_source") == "torchscript":
        models_used.append("CT-Brain-Volume-Segmentation")

    heur = run_ct_brain_heuristics(
        volume,
        meta,
        vol_info.get("ventricle_volume_ml"),
        enabled_ncc=ct_brain_config.CT_BRAIN_NCC_ENABLED,
        enabled_midline=ct_brain_config.CT_BRAIN_MIDLINE_ENABLED,
    )
    pathology_scores.update({k: v for k, v in heur.items() if k not in pathology_scores})

    findings: list[Finding] = []
    from vista3d_integration import enrich_vista3d_metadata

    enrich_vista3d_metadata(
        volume=volume,
        meta=meta,
        film_photo=film_photo,
        degraded=degraded,
        pathology_scores=pathology_scores,
        findings=findings,
    )

    if film_photo:
        findings.append(
            Finding(
                label="Film photo input (mobile photos of printed CT/MRI film)",
                description=(
                    "Analysis is based on phone photographs of printed films, not DICOM. "
                    "ICH scores, subtypes, volumes, and heuristics are approximate only — obtain original CT data when possible."
                ),
                severity="warning",
                confidence=100.0,
                region="Brain",
            )
        )

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

    if subtype_scores:
        top = max(subtype_scores.items(), key=lambda kv: kv[1])
        if top[1] >= 0.35:
            findings.append(
                Finding(
                    label="ICH subtype distribution (auxiliary model)",
                    description=f"Highest subtype score: {top[0]}={top[1]:.2f}. Correlation required.",
                    severity="warning" if top[1] < 0.6 else "critical",
                    confidence=min(95.0, 40.0 + top[1] * 55.0),
                    region="Brain",
                )
            )

    if heur.get("midline_shift_mm") is not None and float(heur["midline_shift_mm"]) > 3.0:
        findings.append(
            Finding(
                label="Midline shift (heuristic)",
                description=f"Estimated midline shift ~{heur['midline_shift_mm']} (units depend on spacing metadata).",
                severity="critical",
                confidence=70.0,
                region="Brain",
            )
        )

    if heur.get("hydrocephalus_flag"):
        findings.append(
            Finding(
                label="Hydrocephalus pattern (volume heuristic)",
                description="Enlarged ventricular volume by segmentation/heuristic threshold.",
                severity="warning",
                confidence=65.0,
                region="Brain",
            )
        )

    if heur.get("ncc_suspect"):
        findings.append(
            Finding(
                label="Calcification pattern — NCC differential (heuristic)",
                description="Multiple small intracranial calcifications detected; correlate for neurocysticercosis vs TB/toxoplasma.",
                severity="warning",
                confidence=55.0,
                region="Brain",
            )
        )

    impression = "NCCT brain analysis complete. Clinical correlation required."
    if inference_mode == "weights_required":
        impression = "NCCT brain received — AI hemorrhage model not configured; no automated ICH score."
    elif inference_mode == "ci_dummy":
        impression = "NCCT brain CI dummy inference — not for clinical use."

    findings_out = [f.model_dump() if isinstance(f, Finding) else f for f in findings]
    
    # Extract representative slices for LLM vision in film-photo mode
    llm_images_b64: list[str] | None = None
    if film_photo and meta:
        try:
            from film_photo_reporting import extract_film_photo_images_for_llm
            llm_images_b64 = extract_film_photo_images_for_llm(
                meta,
                max_images=10,
                min_quality_threshold=30.0,
            )
            if llm_images_b64:
                logger.info(
                    "CT brain film-photo mode: extracted %d slices for LLM visual interpretation",
                    len(llm_images_b64),
                )
        except Exception as e:
            logger.warning("Failed to extract film-photo slices for CT brain LLM: %s", e)
            llm_images_b64 = None
    
    narrative, narr_tags = _call_ct_brain_narrative(
        impression=impression,
        findings=findings_out,
        pathology_scores=pathology_scores,
        patient_context=patient_context,
        film_photo=film_photo,
        image_b64_list=llm_images_b64,
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

    conf = _derive_confidence(
        inference_mode,
        ich_prob,
        bool(subtype_scores),
        vol_info.get("segmentation_source"),
    )
    disc = DISCLAIMER
    if film_photo:
        apply_film_photo_pathology_scores(pathology_scores)
        attach_film_meta_to_structures(structures, meta)
        conf = cap_confidence_for_film(str(conf))
        disc = merge_disclaimer_with_film(
            DISCLAIMER, True, FILM_PHOTO_DISCLAIMER_ADDENDUM
        )

    return {
        "modality": "ct_brain",
        "findings": findings_out,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": conf,
        "models_used": models_used,
        "disclaimer": disc,
    }
