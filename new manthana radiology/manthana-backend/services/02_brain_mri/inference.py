"""
Manthana — Brain MRI Inference
TotalSegmentator total_mr + SynthSeg (subprocess) + optional Prima pipeline + lesion segmentation.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Resolve shared package before service-local preprocessing.py shadows it (Docker + local dev).
_BRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_SHARED = os.path.normpath(os.path.join(_BRAIN_DIR, "..", "..", "shared"))


def _ensure_shared_path_first() -> None:
    """Insert backend `shared/` at sys.path[0] even if it already appears later (e.g. e2e scripts)."""
    for _p in (_BACKEND_SHARED, "/app/shared"):
        if not os.path.isdir(_p):
            continue
        while _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)


_ensure_shared_path_first()
from brain_heuristics import run_brain_heuristics
from disclaimer import DISCLAIMER, FILM_PHOTO_DISCLAIMER_ADDENDUM
from film_photo_reporting import (
    FILM_PHOTO_NARRATIVE_PREFIX,
    apply_film_photo_pathology_scores,
    attach_film_meta_to_structures,
    cap_confidence_for_film,
    merge_disclaimer_with_film,
)
from lesion_segmentation import run_lesion_segmentation
from prima_pipeline import run_prima_study, is_prima_available
from schemas import Finding
from synthseg_runner import run_synthseg
from totalseg_runner import get_totalseg_version, run_totalseg, structure_list_from_result
from wmh_analysis import run_wmh_analysis

logger = logging.getLogger("manthana.brain_mri")

PIPELINE_VERSION = "manthana-brain-mri-v1"

_BRAIN_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_BRAIN_MRI_SYSTEM = _BRAIN_PROMPT_DIR / "brain_mri_system.md"

# Report / narrative context (no LLM required — deterministic stub for assembly + e2e)
INDIA_CLINICAL_NOTE = (
    "India-relevant differentials for focal brain lesions or seizures include neurocysticercosis (NCC), "
    "tuberculoma/TB meningitis, pyogenic abscess, and Japanese encephalitis in endemic seasons — correlate "
    "with travel, vaccination, CSF, and contrast imaging. Stroke and glioma remain universal differentials."
)


def _mri_narrative_policy() -> str:
    """off = no LLM narrative; anything else = OpenRouter (SSOT: config/cloud_inference.yaml)."""
    v = (os.environ.get("MRI_NARRATIVE_POLICY", "openrouter") or "openrouter").strip().lower()
    if v == "off":
        return "off"
    return "openrouter"


def _mri_narrative_vision_enabled() -> bool:
    return os.environ.get("MRI_NARRATIVE_VISION", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _device_for_totalseg() -> str:
    dev = (os.getenv("DEVICE") or "").strip().lower()
    if dev == "cpu":
        return "cpu"
    return os.getenv("TOTALSEG_DEVICE", "gpu")


def _float_pathology_scores(scores: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (scores or {}).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _first_volume_file_in_dir(d: str) -> str | None:
    import glob

    d = os.path.abspath(d)
    for pattern in ("*.nii.gz", "*.nii", "*.dcm", "*.dic"):  # Added .dic extension
        paths = sorted(glob.glob(os.path.join(d, pattern)))
        for p in paths:
            if os.path.isfile(p):
                return p
    return None


def _infer_emergency_flags(pathology_scores: dict, findings: list[Finding]) -> list[str]:
    flags: list[str] = []
    ms = pathology_scores.get("midline_shift_mm")
    try:
        if ms is not None and float(ms) > 5.0:
            flags.append("midline_shift")
    except (TypeError, ValueError):
        pass
    for f in findings:
        text = f"{f.label} {f.description or ''}".lower()
        if "midline" in text and "shift" in text:
            flags.append("midline_shift")
        if "herniation" in text:
            flags.append("herniation")
    return list(dict.fromkeys(flags))


def _build_narrative_report(
    clinical_notes: str,
    impression: str,
    pathology_scores: dict,
    findings: list[Finding],
) -> str:
    """Deterministic narrative stub (Kimi/LLM can replace upstream in report_assembly)."""
    parts = [
        "BRAIN MRI — STRUCTURED SUMMARY (AI-assisted; not a standalone report)",
        "",
        f"Impression: {impression}",
        "",
        INDIA_CLINICAL_NOTE,
        "",
        "Key automated metrics (when available): coarse brain volume (total_mr brain_cm3), SynthSeg QC/volumes, "
        "Prima logits — see pathology_scores and structures.",
    ]
    if clinical_notes:
        parts.extend(["", f"Clinical context provided: {clinical_notes[:1500]}", ""])
    if pathology_scores:
        top = list(pathology_scores.items())[:6]
        parts.append("Selected scores: " + "; ".join(f"{k}={v:.4g}" for k, v in top if isinstance(v, (int, float))))
    parts.append("")
    parts.append(
        "Emergency: review urgently if midline shift, herniation signs, or acute infarct pattern on source imaging — "
        "this pipeline does not replace urgent neurosurgical evaluation."
    )
    return "\n".join(parts)


def _read_brain_mri_system_prompt(prompt_path: Path | None = None) -> str:
    p = prompt_path or _BRAIN_MRI_SYSTEM
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8")[:16000]
    except OSError:
        pass
    return (
        "You are a senior neuroradiologist. Summarise from JSON and scores only. "
        "India: NCC, TB, JE, malaria, stroke epidemiology; EMERGENCY flags per user instructions."
    )


def _parse_patient_context_from_notes(notes: str) -> dict:
    """Parse patient_context_json merged after clinical_notes (often last JSON object in the string)."""
    s = (notes or "").strip()
    if not s:
        return {}
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else {}
    except json.JSONDecodeError:
        pass
    for line in reversed(s.splitlines()):
        line = line.strip()
        if len(line) < 2 or not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
            if isinstance(o, dict):
                return o
        except json.JSONDecodeError:
            continue
    return {}


def _volume_middle_axial_b64_png(vol: np.ndarray) -> str | None:
    """Middle slice along the dominant slice axis (same heuristic as aortic diameter)."""
    try:
        v = np.asarray(vol)
        if v.ndim != 3:
            return None
        slice_axis = 2 if v.shape[2] > 1 else (0 if v.shape[0] > 1 else 1)
        if v.shape[slice_axis] < 1:
            return None
        z = int(v.shape[slice_axis] // 2)
        sl = np.take(v, z, axis=slice_axis).astype(np.float32, copy=False)
        mn, mx = float(sl.min()), float(sl.max())
        if mx > mn:
            sl = (sl - mn) / (mx - mn) * 255.0
        else:
            sl = np.zeros_like(sl)
        u8 = np.clip(sl, 0, 255).astype(np.uint8)
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(u8, mode="L").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.debug("MRI middle-slice PNG base64 failed: %s", e)
        return None


def _mri_narrative_user_text(
    pathology_scores: dict,
    impression: str,
    patient_context: dict,
    findings: list[dict],
) -> str:
    scores_json = json.dumps(pathology_scores, indent=2)[:12000]
    patient_json = json.dumps(patient_context or {}, indent=2)[:4000]
    findings_json = json.dumps(findings, indent=2)[:8000]

    ps = pathology_scores or {}
    brain_vol = ps.get("brain_vol_cm3")
    if brain_vol is None:
        brain_vol = ps.get("brain_cm3")
    hip_l = ps.get("hippocampus_left_cm3")
    hip_r = ps.get("hippocampus_right_cm3")
    ms = ps.get("midline_shift_mm")
    wm = ps.get("wm_hyperintensity_cm3")

    ctx = patient_context or {}
    age = ctx.get("age", ctx.get("patient_age", ""))
    sex = ctx.get("sex", ctx.get("gender", ""))
    ch = ctx.get("clinical_history", ctx.get("history", ""))
    symptoms = ctx.get("symptoms", ctx.get("presenting_complaint", ""))
    region = ctx.get("geographic_region", ctx.get("location_body", ""))

    india_block = (
        "India-specific context (weave into differential where appropriate):\n"
        "- NCC (neurocysticercosis) — leading ring-enhancing lesion cause in India; endemic in Bihar, UP, Rajasthan.\n"
        "- TB meningitis / tuberculoma — second commonest cause of ring-enhancing lesions.\n"
        "- Japanese encephalitis — seasonal; Bihar/UP/Assam belts.\n"
        "- Cerebral malaria (P. falciparum) — northeastern states.\n"
        "- Stroke — Indians often present ~15 years earlier than Western cohorts (mean age ~58 vs ~73); "
        "hypertension is a main risk factor.\n"
        "EMERGENCY flags (must use critical severity language when metrics/findings support):\n"
        "- Midline shift >5 mm → neurosurgery URGENT.\n"
        "- Uncal herniation → EMERGENCY.\n"
        "- Acute large infarct >1/3 MCA territory → thrombolysis window considerations.\n"
    )

    summary = (
        "TotalSegmentator / pipeline summary (numeric):\n"
        f"  Brain volume: {brain_vol if brain_vol is not None else 'N/A'} cm³\n"
        f"  Left hippocampus: {hip_l if hip_l is not None else 'N/A'} cm³\n"
        f"  Right hippocampus: {hip_r if hip_r is not None else 'N/A'} cm³\n"
        f"  Midline shift: {ms if ms is not None else 'N/A'} mm\n"
        f"  WM hyperintensity volume: {wm if wm is not None else 'N/A'} cm³\n"
    )

    return (
        f"{summary}\n"
        f"FULL pathology_scores JSON:\n{scores_json}\n\n"
        f"IMPRESSION (pipeline):\n{impression}\n\n"
        f"STRUCTURED_FINDINGS:\n{findings_json}\n\n"
        f"Patient: {age}y {sex}, region: {region or 'not stated'}\n"
        f"Presenting symptoms / complaint: {symptoms or 'not stated'}\n"
        f"Clinical history (including risk factors): {ch or 'not stated'}\n\n"
        f"PATIENT_CONTEXT JSON:\n{patient_json}\n\n"
        f"{india_block}\n\n"
        "Generate a concise brain MRI radiology-style narrative with India-specific differentials. "
        "Ground every statement in the JSON scores and findings; do not invent lesions or numbers."
    )


def _call_mri_narrative(
    *,
    pathology_scores: dict,
    patient_context: dict,
    image_b64: str | None,
    image_b64_list: list[str] | None = None,
    prompt_path: Path,
    impression: str,
    findings: list[dict],
    film_photo: bool = False,
) -> tuple[str, list[str], dict]:
    """
    MRI_NARRATIVE_POLICY: off | openrouter (default). Legacy Kimi/Anthropic values are treated as on.
    MRI_NARRATIVE_VISION: when disabled, text-only OpenRouter call.
    Uses complete_with_schema for schema enforcement and contradiction detection.
    
    For film_photo mode: image_b64_list contains 4-15 representative slices for multi-image vision.
    Never raises; returns ("", [], {}) if all fail or policy is off.
    """
    if _mri_narrative_policy() == "off":
        return "", [], {}

    system = _read_brain_mri_system_prompt(prompt_path)
    if film_photo:
        system = FILM_PHOTO_NARRATIVE_PREFIX + system
    user_text = _mri_narrative_user_text(pathology_scores, impression, patient_context, findings)
    tag = "OpenRouter-narrative-MRI"

    try:
        from llm_router import llm_router
    except Exception as exc:
        logger.warning("MRI narrative: llm_router unavailable: %s", exc)
        return "", [], {}

    # Determine vision mode: multi-image for film-photo, single image otherwise
    has_multi_image = image_b64_list and len(image_b64_list) > 0 and _mri_narrative_vision_enabled()
    has_single_image = image_b64 and _mri_narrative_vision_enabled()

    # Use complete_with_schema for schema enforcement and contradiction detection
    try:
        if has_multi_image:
            # Film-photo mode: send multiple representative slices
            logger.info("MRI narrative: using multi-image vision with %d film-photo slices", len(image_b64_list))
            out = llm_router.complete_with_schema(
                "narrative_brain_mri",
                system,
                user_text,
                modality="brain_mri",
                image_b64_list=image_b64_list,
                image_mime="image/png",
                max_tokens=2000,  # Increased for multi-image analysis
                automated_scores=pathology_scores,
            )
        elif has_single_image:
            out = llm_router.complete_with_schema(
                "narrative_brain_mri",
                system,
                user_text,
                modality="brain_mri",
                image_b64=image_b64,
                image_mime="image/png",
                max_tokens=1600,
                automated_scores=pathology_scores,
            )
        else:
            out = llm_router.complete_with_schema(
                "narrative_brain_mri",
                system,
                user_text,
                modality="brain_mri",
                max_tokens=1600,
                automated_scores=pathology_scores,
            )
        txt = (out.get("content") or "").strip()
        if txt:
            # Log schema validation and contradiction status
            if not out.get("schema_valid"):
                logger.warning("MRI narrative schema validation failed: %s", out.get("schema_error"))
            contradiction = out.get("contradiction_check", {})
            if contradiction.get("has_contradictions"):
                logger.warning("MRI narrative contradictions detected: %d critical, %d warnings",
                    contradiction.get("critical_count", 0), contradiction.get("warning_count", 0))
            return txt, [tag], out
    except Exception as exc:
        logger.warning("OpenRouter MRI schema narrative failed: %s", exc)
        # Fallback to plain complete_for_role
        try:
            if has_multi_image:
                out = llm_router.complete_for_role(
                    "narrative_brain_mri",
                    system,
                    user_text,
                    image_b64_list=image_b64_list,
                    image_mime="image/png",
                    max_tokens=2000,
                )
            elif has_single_image:
                out = llm_router.complete_for_role(
                    "narrative_brain_mri",
                    system,
                    user_text,
                    image_b64=image_b64,
                    image_mime="image/png",
                    max_tokens=1600,
                )
            else:
                out = llm_router.complete_for_role(
                    "narrative_brain_mri",
                    system,
                    user_text,
                    max_tokens=1600,
                )
            txt = (out.get("content") or "").strip()
            if txt:
                return txt, [tag], {}
        except Exception as exc2:
            logger.warning("OpenRouter MRI fallback narrative failed: %s", exc2)
    return "", [], {}


def is_loaded() -> dict:
    """Two-tier readiness: TotalSeg required for ready; SynthSeg/Prima optional."""
    totalseg_ok = False
    try:
        import totalsegmentator  # noqa: F401

        totalseg_ok = True
    except ImportError:
        pass
    synthseg_ok = os.path.isfile(
        os.getenv("SYNTHSEG_SCRIPT", "/opt/SynthSeg/scripts/commands/SynthSeg_predict.py")
    )
    prima_ok = is_prima_available()  # Use actual availability check, not just config file
    ready = totalseg_ok
    full = totalseg_ok and synthseg_ok and prima_ok
    return {
        "totalseg": totalseg_ok,
        "synthseg": synthseg_ok,
        "prima": prima_ok,
        "prima_configured": bool(os.getenv("PRIMA_CONFIG_YAML", "").strip()),
        "prima_weights_present": prima_ok,
        "ready": ready,
        "full": full,
    }


def run_brain_mri_pipeline(
    filepath: str,
    job_id: str,
    series_dir: str = "",
    clinical_notes: str = "",
) -> dict:
    logger.info(f"[{job_id}] Running brain MRI pipeline...")
    notes = (clinical_notes or "").strip()

    vol_film, meta_film = _try_load_film_photo_volume(filepath, series_dir)
    film_photo = bool(meta_film and meta_film.get("film_photo_mode"))
    if vol_film is not None:
        volume = vol_film
    else:
        volume = _load_brain_volume(filepath)

    degraded_2d = False if film_photo else _is_degraded_input(volume, filepath)

    totalseg_in = _totalseg_input_path(filepath, series_dir, job_id)
    nifti_for_synth = _resolve_nifti_for_synthseg(filepath, series_dir, job_id)
    if (not nifti_for_synth or not os.path.isfile(nifti_for_synth)) and totalseg_in.lower().endswith(
        (".nii", ".nii.gz")
    ):
        nifti_for_synth = totalseg_in

    tot_names, tot_volumes, tot_ok = _run_totalseg_mr(totalseg_in, volume, film_photo=film_photo)
    pathology_scores: dict = {}
    pathology_scores.update(tot_volumes)

    synth = {"available": False}
    if nifti_for_synth and os.path.isfile(nifti_for_synth):
        synth = run_synthseg(nifti_for_synth, job_id)
        if synth.get("available") and synth.get("volumes"):
            for k, v in synth["volumes"].items():
                pathology_scores[f"synthseg_{k}"] = float(v)
            if synth.get("qc_score") is not None:
                pathology_scores["synthseg_qc_score"] = float(synth["qc_score"])

    prima = run_prima_study(series_dir or filepath, job_id)
    if prima.get("available") and prima.get("scores"):
        for k, v in prima["scores"].items():
            pathology_scores[f"prima_{k}"] = float(v)

    # Run new P5 modules: lesion segmentation, WMH analysis, brain heuristics
    lesion = run_lesion_segmentation(volume)
    if lesion.get("available"):
        pathology_scores["lesion_tumor_volume_ml"] = lesion.get("tumor_volume_ml")
        pathology_scores["lesion_enhancing_volume_ml"] = lesion.get("enhancing_volume_ml")
        pathology_scores["lesion_edema_volume_ml"] = lesion.get("edema_volume_ml")
        pathology_scores["lesion_location"] = lesion.get("lesion_location")

    wmh = run_wmh_analysis(volume, synthseg_volumes=synth.get("volumes"))
    if wmh.get("available"):
        pathology_scores["wmh_volume_ml"] = wmh.get("wmh_volume_ml")
        pathology_scores["fazekas_score"] = wmh.get("fazekas_score")
        pathology_scores["wmh_distribution"] = wmh.get("distribution")

    # Get ventricle and brain volumes for heuristics
    vent_vol = synth.get("volumes", {}).get("ventricles", 0) if synth.get("volumes") else 0
    brain_vol = synth.get("volumes", {}).get("brain", 0) if synth.get("volumes") else 0
    brain_heur = run_brain_heuristics(
        volume=volume,
        synthseg_volumes=synth.get("volumes"),
        ventricle_volume_ml=vent_vol if vent_vol else None,
        brain_volume_ml=brain_vol if brain_vol else None,
    )
    bh = brain_heur.get("hydrocephalus", {})
    if bh.get("evans_index_proxy") is not None:
        pathology_scores["hydrocephalus_suspected"] = float(bh.get("hydrocephalus_suspected", False))
        pathology_scores["evans_index_proxy"] = bh.get("evans_index_proxy")
    ba = brain_heur.get("atrophy", {})
    if ba.get("hippocampal_ratio") is not None:
        pathology_scores["hippocampal_ratio"] = ba.get("hippocampal_ratio")
        pathology_scores["atrophy_detected"] = float(ba.get("atrophy_detected", False))
    btb = brain_heur.get("tb_meningitis", {})
    if btb.get("tb_meningitis_suspected") is not None:
        pathology_scores["tb_meningitis_suspected"] = float(btb.get("tb_meningitis_suspected", False))
        pathology_scores["tb_meningitis_score"] = btb.get("score")
    bnc = brain_heur.get("ncc", {})
    if bnc.get("ncc_suspected") is not None:
        pathology_scores["ncc_suspected"] = float(bnc.get("ncc_suspected", False))
        pathology_scores["ncc_ring_lesion_count"] = bnc.get("ring_lesion_count", 0)

    findings = _build_brain_findings(
        tot_ok=tot_ok,
        tot_names=tot_names,
        synth=synth,
        prima=prima,
        lesion=lesion,
        wmh=wmh,
        brain_heur=brain_heur,
        degraded_2d=degraded_2d,
        filepath=filepath,
        clinical_notes=notes,
        film_photo=film_photo,
    )
    impression = _build_impression(pathology_scores, tot_ok, synth, prima, lesion, wmh)
    models_used = _models_used(tot_ok, synth, prima, lesion, wmh, film_photo=film_photo)

    totalseg_ok = tot_ok
    synthseg_ok = bool(synth.get("available"))
    prima_ok = bool(prima.get("available"))
    confidence = (
        "high"
        if prima_ok and (totalseg_ok or synthseg_ok)
        else "medium"
        if totalseg_ok or synthseg_ok
        else "low"
    )

    structures: dict = {
        "segment_names": tot_names,
        "totalseg_volumes_cm3": tot_volumes,
        "totalseg_available": tot_ok,
        "synthseg": {k: v for k, v in synth.items() if k != "volumes"},
        "synthseg_available": bool(synth.get("available")),
        "prima_available": bool(prima.get("available")),
        "lesion_segmentation": {k: v for k, v in lesion.items() if k not in ("tumor_volume_ml", "enhancing_volume_ml", "edema_volume_ml")},
        "lesion_available": bool(lesion.get("available")),
        "wmh_analysis": {k: v for k, v in wmh.items() if k != "wmh_volume_ml"},
        "wmh_available": bool(wmh.get("available")),
        "brain_heuristics": brain_heur,
        "clinical_notes": notes,
        "series_dir_used": bool(series_dir and os.path.isdir(series_dir)),
        "algorithm_version": {
            "pipeline_version": PIPELINE_VERSION,
            "totalsegmentator": get_totalseg_version(),
            "narrative_policy": _mri_narrative_policy(),
            "mri_narrative_vision": _mri_narrative_vision_enabled(),
        },
    }
    if synth.get("volumes"):
        structures["synthseg_volumes_cm3"] = synth["volumes"]

    pathology_scores = _float_pathology_scores(pathology_scores)
    findings_payload = [f.model_dump() if isinstance(f, Finding) else dict(f) for f in findings]
    patient_ctx = _parse_patient_context_from_notes(notes)
    _pol = _mri_narrative_policy()
    need_vision_slice = _mri_narrative_vision_enabled() and _pol != "off"
    
    # Determine vision input: multi-image for film-photo mode, single slice for normal mode
    llm_images_b64: list[str] | None = None
    slice_b64: str | None = None
    
    if film_photo and need_vision_slice and meta_film:
        # Film-photo mode: extract 4-10 representative slices for multi-image vision
        try:
            from preprocessing.film_photo_loader import extract_representative_slices_for_llm
            # Limit to 10 images for cost control, minimum 4 for meaningful coverage
            llm_images_b64 = extract_representative_slices_for_llm(
                meta_film, 
                max_images=10,
                min_quality_threshold=30.0  # Lower threshold for film photos
            )
            if llm_images_b64:
                logger.info(
                    "Film-photo mode: sending %d representative slices to LLM for visual interpretation",
                    len(llm_images_b64)
                )
        except Exception as e:
            logger.warning("Failed to extract film-photo slices for LLM vision: %s", e)
            llm_images_b64 = None
    
    if not llm_images_b64 and need_vision_slice:
        # Normal mode: single middle slice
        slice_b64 = _volume_middle_axial_b64_png(volume)
    
    llm_narr, narr_tags, _ = _call_mri_narrative(
        pathology_scores=pathology_scores,
        patient_context=patient_ctx,
        image_b64=slice_b64,
        image_b64_list=llm_images_b64,
        prompt_path=_BRAIN_MRI_SYSTEM,
        impression=impression,
        findings=findings_payload,
        film_photo=film_photo,
    )
    if llm_narr.strip():
        structures["narrative_report"] = llm_narr.strip()
        for t in narr_tags:
            if t not in models_used:
                models_used.append(t)
    else:
        structures["narrative_report"] = _build_narrative_report(
            notes, impression, pathology_scores, findings
        )
    structures["emergency_flags"] = _infer_emergency_flags(pathology_scores, findings)
    structures["india_note"] = INDIA_CLINICAL_NOTE

    findings_out: list[dict] = []
    for f in findings:
        findings_out.append(f.model_dump() if isinstance(f, Finding) else dict(f))

    for _tmp in {nifti_for_synth, totalseg_in}:
        if (
            _tmp
            and _tmp != filepath
            and _tmp.startswith(tempfile.gettempdir())
            and os.path.isfile(_tmp)
        ):
            try:
                os.unlink(_tmp)
            except OSError:
                pass

    disc = DISCLAIMER
    if film_photo:
        apply_film_photo_pathology_scores(pathology_scores)
        attach_film_meta_to_structures(structures, meta_film)
        confidence = cap_confidence_for_film(str(confidence))
        disc = merge_disclaimer_with_film(DISCLAIMER, True, FILM_PHOTO_DISCLAIMER_ADDENDUM)

    return {
        "job_id": job_id,
        "modality": "brain_mri",
        "findings": findings_out,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": confidence,
        "models_used": models_used,
        "disclaimer": disc,
    }


def _is_degraded_input(volume: np.ndarray, _filepath_unused: str = "") -> bool:
    """Check if volume is degraded (2D or single slice). filepath param kept for API compat but unused."""
    v = np.asarray(volume)
    if v.ndim < 3:
        return True
    if v.ndim == 3 and min(v.shape) <= 1:
        return True
    return False


def _totalseg_input_path(filepath: str, series_dir: str, job_id: str) -> str:
    """TotalSegmentator expects NIfTI; convert DICOM series directory when needed."""
    pl = (filepath or "").lower()
    if pl.endswith(".nii") or pl.endswith(".nii.gz"):
        return filepath
    if series_dir and os.path.isdir(series_dir):
        try:
            from preprocessing.dicom_utils import dicom_series_to_nifti

            out = os.path.join(tempfile.gettempdir(), f"{job_id}_totalseg_series.nii.gz")
            p = dicom_series_to_nifti(series_dir, output_path=out)
            if p and os.path.isfile(p):
                return p
        except Exception as e:
            logger.warning("DICOM series→NIfTI for TotalSeg failed: %s", e)
    return filepath


def _resolve_nifti_for_synthseg(filepath: str, series_dir: str, job_id: str) -> str | None:
    """Return path to NIfTI for SynthSeg, or None if not available."""
    pl = filepath.lower()
    if pl.endswith(".nii") or pl.endswith(".nii.gz"):
        return filepath
    if series_dir and os.path.isdir(series_dir):
        try:
            from preprocessing.dicom_utils import dicom_series_to_nifti

            out = os.path.join(tempfile.gettempdir(), f"{job_id}_synthseg_in.nii.gz")
            return dicom_series_to_nifti(series_dir, output_path=out)
        except Exception as e:
            logger.warning("DICOM series → NIfTI for SynthSeg failed: %s", e)
            return None
    return None


def _load_brain_volume(filepath: str) -> np.ndarray:
    ext = filepath.lower().split(".")[-1]
    if ext in ("nii", "gz"):
        from preprocessing.nifti_utils import read_nifti

        vol, _ = read_nifti(filepath)
        return vol
    elif ext == "dcm":
        from preprocessing.dicom_utils import read_dicom

        arr, _ = read_dicom(filepath)
        return arr
    else:
        from preprocessing.image_utils import load_image, to_grayscale

        return to_grayscale(load_image(filepath))


def _try_load_film_photo_volume(filepath: str, series_dir: str) -> tuple[np.ndarray | None, dict | None]:
    """Load stacked film photos from a directory (series_dir or filepath when it is a dir)."""
    from preprocessing.film_photo_loader import (
        discover_raster_files,
        is_film_photo_input,
        load_film_photos_as_volume,
    )

    for d in ((series_dir or "").strip(), filepath or ""):
        if not d or not os.path.isdir(d):
            continue
        if not is_film_photo_input(d):
            continue
        vol, meta = load_film_photos_as_volume(discover_raster_files(d))
        if isinstance(meta, dict):
            meta.setdefault("modality", "MR")
        return vol, meta
    return None, None


def _run_totalseg_mr(
    filepath: str, volume: np.ndarray, *, film_photo: bool = False
) -> tuple[list, dict, bool]:
    if film_photo:
        return [], {}, False
    pl = (filepath or "").lower()
    if pl.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        return [], {}, False
    try:
        inp = filepath if filepath and os.path.isfile(filepath) else volume
        result = run_totalseg(
            inp,
            task="total_mr",
            fast=True,
            device=_device_for_totalseg(),
        )
        names = structure_list_from_result(result)
        vols = result.get("volumes_cm3") or {}
        return names, vols, True
    except Exception as e:
        logger.warning("TotalSeg MRI failed: %s", e)
        return [], {}, False


def _build_brain_findings(
    tot_ok: bool,
    tot_names: list,
    synth: dict,
    prima: dict,
    lesion: dict,
    wmh: dict,
    brain_heur: dict,
    degraded_2d: bool,
    filepath: str,
    clinical_notes: str,
    film_photo: bool = False,
) -> list[Finding]:
    out: list[Finding] = []
    pl = (filepath or "").lower()

    if film_photo:
        out.append(
            Finding(
                label="Film photo input (mobile photos of printed CT/MRI film)",
                description=(
                    "This analysis is based on mobile phone photographs of printed MRI films, NOT original DICOM. "
                    "Volumetric measurements, segmentation accuracy, and quantitative scores are significantly less reliable. "
                    "For definitive diagnosis, obtain original DICOM from the imaging centre."
                ),
                severity="warning",
                confidence=100.0,
                region="Brain",
            )
        )

    if not film_photo and pl.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        out.append(
            Finding(
                label="2D image input",
                description="Volumetric brain segmentation requires NIfTI or DICOM series. 2D raster input cannot be segmented with TotalSeg/SynthSeg.",
                severity="warning",
                confidence=100.0,
                region="Brain",
            )
        )
    elif degraded_2d:
        out.append(
            Finding(
                label="Limited volumetric data",
                description="Input appears to be a single slice or non-volumetric array. Full brain MRI analysis is degraded.",
                severity="warning",
                confidence=90.0,
                region="Brain",
            )
        )

    if tot_ok and tot_names:
        out.append(
            Finding(
                label="TotalSegmentator MRI (whole-body task)",
                description=f"Segmentation labels present: {len(tot_names)} structures. total_mr includes a single coarse brain label — use SynthSeg for brain sub-structure volumes when available.",
                severity="info",
                confidence=85.0,
                region="Brain",
            )
        )
    elif not tot_ok:
        out.append(
            Finding(
                label="TotalSegmentator MRI unavailable",
                description="total_mr segmentation did not complete successfully.",
                severity="warning",
                confidence=100.0,
                region="Brain",
            )
        )

    if synth.get("available"):
        qc = synth.get("qc_score")
        qc_note = ""
        if qc is not None and qc < 0.5:
            qc_note = f" QC score {qc:.2f} suggests unreliable segmentation."
        out.append(
            Finding(
                label="SynthSeg brain parcellation",
                description=f"Brain structure volumes extracted (SynthSeg).{qc_note}",
                severity="warning" if (qc is not None and qc < 0.5) else "info",
                confidence=88.0,
                region="Brain",
            )
        )
    else:
        out.append(
            Finding(
                label="SynthSeg unavailable",
                description=f"Brain sub-structure volumes not computed ({synth.get('reason', 'not installed or failed')}).",
                severity="info",
                confidence=100.0,
                region="Brain",
            )
        )

    if prima.get("available"):
        out.append(
            Finding(
                label="Prima study classification",
                description="Prima diagnostic head produced scores.",
                severity="info",
                confidence=70.0,
                region="Brain",
            )
        )
    else:
        out.append(
            Finding(
                label="Prima classification unavailable",
                description=prima.get("reason", "prima_not_configured"),
                severity="info",
                confidence=100.0,
                region="Brain",
            )
        )

    # New P5 findings
    if lesion.get("available"):
        tumor_vol = lesion.get("tumor_volume_ml", 0) or 0
        sev = "critical" if tumor_vol > 50 else ("warning" if tumor_vol > 10 else "info")
        out.append(
            Finding(
                label="Lesion segmentation",
                description=f"Brain lesion segmentation: tumor {tumor_vol:.1f} ml, edema {lesion.get('edema_volume_ml', 0) or 0:.1f} ml at {lesion.get('lesion_location', 'unknown')}",
                severity=sev,
                confidence=75.0,
                region="Brain",
            )
        )

    if wmh.get("available"):
        faz = wmh.get("fazekas_score", 0) or 0
        sev = "warning" if faz >= 2 else "info"
        out.append(
            Finding(
                label="WMH analysis",
                description=f"White matter hyperintensity: Fazekas {faz} ({wmh.get('distribution', 'unknown')}), volume {wmh.get('wmh_volume_ml', 0) or 0:.1f} ml",
                severity=sev,
                confidence=70.0,
                region="Brain",
            )
        )

    # Brain heuristics findings
    bh = brain_heur.get("hydrocephalus", {})
    if bh.get("hydrocephalus_suspected"):
        out.append(
            Finding(
                label="Hydrocephalus suspected",
                description=f"Ventricle:brain ratio {bh.get('evans_index_proxy', 'unknown')} suggests hydrocephalus pattern",
                severity="warning",
                confidence=60.0,
                region="Brain",
            )
        )

    ba = brain_heur.get("atrophy", {})
    if ba.get("atrophy_detected"):
        out.append(
            Finding(
                label="Atrophy pattern detected",
                description=f"Hippocampal ratio {ba.get('hippocampal_ratio', 'unknown')}, pattern: {ba.get('pattern', 'unknown')}",
                severity="info",
                confidence=55.0,
                region="Brain",
            )
        )

    btb = brain_heur.get("tb_meningitis", {})
    if btb.get("tb_meningitis_suspected"):
        out.append(
            Finding(
                label="TB meningitis pattern suspected",
                description="Hydrocephalus with basal cistern changes - correlate with TB clinical history and CSF analysis",
                severity="warning",
                confidence=50.0,
                region="Brain",
            )
        )

    bnc = brain_heur.get("ncc", {})
    if bnc.get("ncc_suspected"):
        count = bnc.get("ring_lesion_count", 0)
        out.append(
            Finding(
                label="NCC pattern suspected",
                description=f"Detected {count} cystic ring lesions - correlate with endemic exposure and serology",
                severity="warning",
                confidence=45.0,
                region="Brain",
            )
        )

    if clinical_notes:
        out.append(
            Finding(
                label="Clinical context noted",
                description=clinical_notes[:2000],
                severity="info",
                confidence=100.0,
                region="Clinical",
            )
        )

    return out


def _build_impression(
    pathology_scores: dict,
    tot_ok: bool,
    synth: dict,
    prima: dict,
    lesion: dict,
    wmh: dict,
) -> str:
    if not tot_ok and not synth.get("available") and not prima.get("available"):
        return "Automated brain MRI analysis could not be completed — clinical correlation and repeat imaging if indicated."
    parts = []
    bc = pathology_scores.get("brain_cm3")
    if bc is not None:
        parts.append(f"Coarse brain volume (total_mr) {float(bc):.1f} cm³.")
    if synth.get("available") and pathology_scores.get("synthseg_qc_score") is not None:
        parts.append(f"SynthSeg QC {float(pathology_scores['synthseg_qc_score']):.2f}.")
    if prima.get("available"):
        parts.append("Prima diagnostic scores available in pathology_scores.")
    if lesion.get("available") and lesion.get("tumor_volume_ml", 0) and lesion["tumor_volume_ml"] > 1:
        parts.append(f"Lesion volume {lesion['tumor_volume_ml']:.1f} ml detected.")
    if wmh.get("available") and wmh.get("fazekas_score", 0) and wmh["fazekas_score"] >= 1:
        parts.append(f"WMH Fazekas {wmh['fazekas_score']}.")
    if not parts:
        return "Brain MRI analysis complete. Clinical correlation recommended."
    return " ".join(parts) + " Clinical correlation recommended."


def _models_used(
    tot_ok: bool,
    synth: dict,
    prima: dict,
    lesion: dict,
    wmh: dict,
    *,
    film_photo: bool = False,
) -> list[str]:
    mu = []
    if film_photo:
        mu.append("Film-photo-stack")
    if tot_ok:
        mu.append("TotalSegmentator-MRI")
    if synth.get("available"):
        mu.append("SynthSeg")
    if prima.get("available"):
        mu.append("Prima")
    if lesion.get("available"):
        mu.append("BraTS-Lesion-Segmentation")
    if wmh.get("available"):
        mu.append("WMH-Fazekas-Analysis")
    return mu


def run_pipeline(
    filepath: str,
    job_id: str,
    patient_context: dict | None = None,
    *,
    series_dir: str = "",
) -> dict:
    """
    E2E-friendly entrypoint: optional patient_context dict (JSON-serialized to clinical_notes),
    and DICOM/NIfTI series directory as filepath (first volume file is selected).
    """
    import json

    notes = ""
    if patient_context:
        notes = json.dumps(patient_context, ensure_ascii=False)
    fp = filepath
    sd = (series_dir or "").strip()
    if os.path.isdir(filepath):
        absd = os.path.abspath(filepath)
        try:
            from preprocessing.film_photo_loader import is_film_photo_input

            if is_film_photo_input(absd):
                return run_brain_mri_pipeline(absd, job_id, series_dir=absd, clinical_notes=notes)
        except Exception:
            pass
        sd = absd
        first = _first_volume_file_in_dir(sd)
        if not first:
            raise ValueError(f"No NIfTI or DICOM file found in directory: {filepath}")
        fp = first

    return run_brain_mri_pipeline(fp, job_id, series_dir=sd, clinical_notes=notes)


def run_brain_mri_pipeline_b64(
    file_b64: str,
    clinical_notes: str = "",
    job_id: str = "",
    filename_hint: str = "",
    series_dir: str = "",
    patient_context_json: str = "",
) -> dict:
    import base64
    import uuid

    merged_notes = (clinical_notes or "").strip()
    pj = (patient_context_json or "").strip()
    if pj:
        merged_notes = f"{merged_notes}\n{pj}".strip() if merged_notes else pj

    try:
        data = base64.b64decode(file_b64, validate=False)
    except Exception as e:
        return {
            "available": False,
            "reason": "invalid_base64",
            "message": str(e),
            "modality": "brain_mri",
        }
    # Default to .nii.gz for NIfTI base64 uploads, fallback to .png only if explicitly image
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
    if filename_hint and filename_hint.lower().endswith(img_exts):
        ext = os.path.splitext(filename_hint)[1]
    else:
        ext = ".nii.gz"  # Default to NIfTI, not PNG
    
    jid = job_id or str(uuid.uuid4())
    upload_dir = "/tmp/manthana_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    fp = os.path.join(upload_dir, f"{jid}_b64{ext or '.nii.gz'}")
    with open(fp, "wb") as f:
        f.write(data)
    try:
        result = run_brain_mri_pipeline(
            fp,
            jid,
            series_dir=series_dir or "",
            clinical_notes=merged_notes,
        )
        result["job_id"] = jid
        return result
    finally:
        try:
            os.unlink(fp)
        except OSError:
            pass
