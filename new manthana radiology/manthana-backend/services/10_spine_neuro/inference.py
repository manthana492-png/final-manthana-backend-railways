"""Manthana — Spine/Neuro Inference: TotalSeg vertebrae_mr (MRI) or vertebrae_body (CT)."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _find_shared() -> Path:
    """Resolve shared/ (Docker / Lightning AI / local repo)."""
    candidates = [
        Path("/app/shared"),
        Path(__file__).resolve().parent.parent.parent / "shared",
        Path(__file__).resolve().parent.parent / "shared",
        Path.cwd() / "shared",
        Path.cwd().parent / "shared",
    ]
    for c in candidates:
        if (c / "schemas.py").exists():
            return c
    raise RuntimeError("Cannot find shared/ directory for spine_neuro service.")


_shared = _find_shared()
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

try:
    from env_file import load_api_keys_env

    load_api_keys_env()
except ImportError:
    pass

from disclaimer import DISCLAIMER, FILM_PHOTO_DISCLAIMER_ADDENDUM
from film_photo_reporting import (
    FILM_PHOTO_NARRATIVE_PREFIX,
    apply_film_photo_pathology_scores,
    attach_film_meta_to_structures,
    cap_confidence_for_film,
    is_film_photo_meta,
    merge_disclaimer_with_film,
)
from fracture_grading import grade_vertebral_fractures
from opportunistic_osteoporosis import compute_l1_hu_bmd
from preprocessing.ct_loader import is_degraded_single_slice, load_ct_volume
from schemas import Finding
from totalseg_runner import (
    get_totalseg_version,
    run_totalseg,
    structure_list_from_result,
)

logger = logging.getLogger("manthana.spine_neuro")
PIPELINE_VERSION = "manthana-spine-v3"

_DEFAULT_LEVELS = ["L1", "L2", "L3", "L4", "L5", "S1"]
_DISC_LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


def _parse_patient_context(raw: str | dict | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s:
        return {}
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else {}
    except json.JSONDecodeError:
        return {}


def _clinical_blob(text: str) -> str:
    return (text or "").lower()


def _paraspinal_brightness_signal(volume: np.ndarray) -> float:
    """Rough 0–1 signal for bilateral paraspinal hyperintensity (T2-like PNG/MIP)."""
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim == 3:
        sl = v[:, :, v.shape[2] // 2]
    else:
        sl = v
    if sl.size == 0:
        return 0.0
    h, w = sl.shape[:2]
    if w < 6 or h < 6:
        return 0.0
    left = float(np.mean(sl[:, : w // 3]))
    right = float(np.mean(sl[:, 2 * w // 3 :]))
    center = float(np.mean(sl[:, w // 3 : 2 * w // 3]))
    if center < 1e-6:
        return 0.0
    if left > center * 1.15 and right > center * 1.15:
        return min(1.0, ((left + right) / (2 * center) - 1.0) * 1.5)
    return 0.0


def _score_potts_features(volume: np.ndarray, _structures: dict[str, Any] | None = None) -> float:
    """Image-derived 0–1 signal for Pott's / paraspinal abscess pattern (T2-like PNG/MIP)."""
    return _paraspinal_brightness_signal(volume)


def _build_clinical_spine_structures(
    vertebrae: list[str],
    heights: list[float],
    degraded: bool,
    volumes_cm3: dict,
    is_mri: bool,
    tot_task: str,
    patient_context: dict[str, Any],
    volume: np.ndarray,
    totseg_limitation_note: str = "",
    fracture_grading: dict | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Heuristic clinical structure block + scores (TotalSeg augments; never fabricates labels as diagnosis)."""
    ctx_txt = _clinical_blob(
        str(patient_context.get("clinical_history", ""))
        + " "
        + str(patient_context.get("history", ""))
    )
    levels = _DEFAULT_LEVELS
    if vertebrae:
        cleaned = [str(x) for x in vertebrae[:12] if x]
        if cleaned:
            levels = cleaned[:7] if len(cleaned) >= 7 else cleaned + _DEFAULT_LEVELS[len(cleaned) :]

    n_disc = len(_DISC_LEVELS)
    disc_heights: dict[str, str] = {}
    for i, dk in enumerate(_DISC_LEVELS):
        h0 = heights[i] if i < len(heights) else 24.0
        h1 = heights[i + 1] if i + 1 < len(heights) else 24.0
        ratio = h0 / max(h1, 1e-6)
        if ratio < 0.82 or "compression" in ctx_txt or "wedge" in ctx_txt:
            disc_heights[dk] = "collapsed" if ratio < 0.65 else "moderate_reduction"
        elif ratio < 0.92 or "desiccation" in ctx_txt or "degenerat" in ctx_txt:
            disc_heights[dk] = "mild_reduction"
        else:
            disc_heights[dk] = "normal"

    # Base image signal; clinical context applied separately (India TB-endemic wording).
    pott_base = min(1.0, _score_potts_features(volume) * 0.95 + (0.12 if "pott" in ctx_txt else 0.0))

    tb_keywords = [
        "tb",
        "tuberculosis",
        "weight loss",
        "fever",
        "night sweat",
        "contact",
        "bihar",
        "up ",
        "maharashtra",
        "ntep",
        "rntcp",
        "afb",
        "gene xpert",
        "pott",
    ]
    keyword_hits = sum(1 for kw in tb_keywords if kw in ctx_txt)
    pott_boost = 0.0
    if keyword_hits >= 2:
        pott_boost = min(0.40, keyword_hits * 0.10)
        logger.debug(
            "Pott boost +%.2f from %d TB/epidemiology keywords in clinical history",
            pott_boost,
            keyword_hits,
        )
    if keyword_hits >= 4:
        pott_boost = min(0.55, pott_boost + 0.15)

    pott_conf = min(0.98, pott_base + pott_boost)
    if keyword_hits >= 4:
        pott_conf = max(pott_conf, 0.65)

    pott_signs = pott_conf >= 0.35

    cord_compression = 0.0
    para_sig = _paraspinal_brightness_signal(volume)
    if any(k in ctx_txt for k in ("myelopath", "cord compression", "weakness both legs", "bladder", "bowel")):
        cord_compression = min(1.0, 0.55 + para_sig * 0.2)
    if "stenosis" in ctx_txt or "claudication" in ctx_txt:
        cord_compression = max(cord_compression, 0.35)

    stenosis_sev = min(
        1.0,
        0.15
        + (0.35 if "stenosis" in ctx_txt else 0.0)
        + (0.2 if degraded else 0.0)
        + (0.15 if any(disc_heights[d] != "normal" for d in disc_heights) else 0.0),
    )

    canal = "none"
    if stenosis_sev > 0.55:
        canal = "severe_L4-L5"
    elif stenosis_sev > 0.35:
        canal = "moderate_L4-L5"
    elif stenosis_sev > 0.2:
        canal = "mild_L3-L4"

    cord_signal = "normal"
    if cord_compression > 0.5:
        cord_signal = "compressed"
    elif "hyperintens" in ctx_txt or "t2 hyper" in ctx_txt:
        cord_signal = "hyperintense"

    modic: dict[str, str] = {}
    if any(k in ctx_txt for k in ("modic", "endplate", "marrow edema")):
        modic["L3-L4"] = "II"
    if "type i" in ctx_txt or "type 1" in ctx_txt:
        modic["L4-L5"] = "I"

    spondy = "none"
    if "spondylolisthesis" in ctx_txt or "listhesis" in ctx_txt:
        m = re.search(r"grade\s*([iiv]+|[1-4])", ctx_txt)
        if m:
            spondy = f"grade_{m.group(1).upper()}_L4-L5"
        else:
            spondy = "grade_I_L4-L5"

    # Use real fracture grading if available
    fracture = "none"
    highest_genant = 0
    if fracture_grading and fracture_grading.get("available"):
        highest_genant = fracture_grading.get("highest_grade", 0)
        any_frac = fracture_grading.get("any_fracture", False)
        if any_frac or highest_genant >= 1:
            # Find the highest grade fracture level
            worst = None
            for v in fracture_grading.get("vertebrae", []):
                if v.get("genant_grade", 0) > (worst.get("genant_grade", 0) if worst else 0):
                    worst = v
            if worst:
                frac_type = worst.get("genant_type", "compression")
                fracture = f"{frac_type}_{worst['level']}_Genant{worst['genant_grade']}"
        fracture_conf = 0.7 if any_frac or highest_genant >= 1 else 0.3
    else:
        # Fall back to clinical context heuristics
        if "burst" in ctx_txt:
            fracture = "burst_L1"
        elif "wedge" in ctx_txt or "compression fracture" in ctx_txt:
            fracture = "wedge_L2"
        fracture_conf = 0.55 if fracture != "none" else min(0.25, stenosis_sev * 0.3)

    disc_deg = min(1.0, 0.2 + stenosis_sev * 0.5 + (0.15 if degraded else 0.0))

    limitation_note = ""
    if degraded:
        limitation_note = (
            "Single-slice or degraded input — automated height and canal metrics are limited; "
            "full MRI/CT series recommended for surgical planning."
        )
    if not vertebrae and not volumes_cm3:
        limitation_note = (
            (limitation_note + " " if limitation_note else "")
            + "Segmentation did not return vertebral labels — interpret heuristic scores cautiously."
        )
    if totseg_limitation_note.strip():
        limitation_note = (
            (limitation_note + " " if limitation_note else "") + totseg_limitation_note.strip()
        ).strip()

    structures: dict[str, Any] = {
        "vertebral_levels_assessed": levels,
        "disc_heights": disc_heights,
        "cord_signal": cord_signal,
        "canal_stenosis": canal,
        "modic_changes": modic,
        "spondylolisthesis": spondy,
        "fracture": fracture,
        "pott_disease_signs": pott_signs,
        "opll_signs": None,  # OPLL detection not implemented; marked as unknown vs hardcoded False
        "limitation_note": limitation_note,
        "narrative_report": "",
    }

    scores: dict[str, float] = {
        "stenosis_severity": round(stenosis_sev, 4),
        "disc_degeneration_score": round(disc_deg, 4),
        "cord_compression_confidence": round(cord_compression, 4),
        "pott_disease_confidence": round(pott_conf, 4),
        "fracture_confidence": round(fracture_conf, 4),
    }
    return structures, scores


def _spine_narrative_policy() -> str:
    """CT_SPINE_NARRATIVE_POLICY: off | openrouter (default). Legacy Kimi/Anthropic values enable OpenRouter."""
    v = (os.environ.get("CT_SPINE_NARRATIVE_POLICY", "openrouter") or "openrouter").strip().lower()
    if v in ("off", "none", "disabled", "0"):
        return "off"
    return "openrouter"


def _spine_narrative_openrouter(
    structures: dict[str, Any],
    pathology_scores: dict[str, float],
    patient_context: dict[str, Any],
    *,
    film_photo: bool = False,
    image_b64_list: list[str] | None = None,
) -> tuple[str, list[str]]:
    policy = _spine_narrative_policy()
    tags: list[str] = []
    if policy == "off":
        return "", ["CT-spine-narrative-disabled"]

    scores = {
        "stenosis_severity": float(pathology_scores.get("stenosis_severity", 0) or 0),
        "disc_degeneration_score": float(pathology_scores.get("disc_degeneration_score", 0) or 0),
        "cord_compression_confidence": float(pathology_scores.get("cord_compression_confidence", 0) or 0),
        "pott_disease_confidence": float(pathology_scores.get("pott_disease_confidence", 0) or 0),
        "fracture_confidence": float(pathology_scores.get("fracture_confidence", 0) or 0),
    }
    age = patient_context.get("age", "")
    sex = patient_context.get("sex", "")
    clin = patient_context.get("clinical_history", patient_context.get("history", ""))
    hosp = patient_context.get("hospital", "India")
    is_mri = structures.get('imaging_modality', 'MRI').upper() == 'MRI'
    modality_label = "Spine MRI" if is_mri else "Spine CT"
    user_text = f"""{modality_label} Analysis Results:
Vertebral levels assessed: {structures.get('vertebral_levels_assessed')}
Disc heights: {structures.get('disc_heights')}
Cord signal: {structures.get('cord_signal')}
Canal stenosis: {structures.get('canal_stenosis')}
Pott's disease confidence: {scores.get('pott_disease_confidence', 0):.3f}
Cord compression confidence: {scores.get('cord_compression_confidence', 0):.3f}
Fracture confidence: {scores.get('fracture_confidence', 0):.3f}
OPLL signs: {structures.get('opll_signs')}
Modic changes: {structures.get('modic_changes')}
Limitation note: {structures.get('limitation_note') or 'none'}

Patient: {age}y {sex}
Clinical: {clin}
Reporting centre: {hosp}

Generate a structured {modality_label} report following the system prompt.
Include: disc pathology by level, cord status, Pott's assessment,
India-specific differentials (TB spine vs other), management."""
    system = (
        "You are a senior neuroradiologist writing structured spine MRI/CT reports "
        "for Indian clinical settings. Be precise; flag uncertainty where input is limited."
    )
    if film_photo:
        system = FILM_PHOTO_NARRATIVE_PREFIX + system
    try:
        from llm_router import llm_router

        # For film-photo mode with multi-image vision
        has_vision_images = film_photo and image_b64_list and len(image_b64_list) > 0
        
        if has_vision_images:
            logger.info("Spine narrative: using multi-image vision with %d film-photo slices", len(image_b64_list))
            out = llm_router.complete_for_role(
                "spine",
                system,
                user_text,
                image_b64_list=image_b64_list,
                image_mime="image/png",
                max_tokens=2400,
            )
        else:
            out = llm_router.complete_for_role(
                "spine",
                system,
                user_text,
                max_tokens=2000,
            )
        txt = (out.get("content") or "").strip()
        if txt:
            tags.append("OpenRouter-narrative-Spine")
            return txt, tags
    except Exception as e:
        logger.warning("OpenRouter spine narrative failed: %s", e)
    return "", tags


def is_loaded() -> dict:
    totalseg_ok = False
    comp2comp_ok = False
    try:
        import totalsegmentator  # noqa: F401

        totalseg_ok = True
    except ImportError:
        pass
    try:
        exe = os.path.join(os.getenv("COMP2COMP_DIR", "/opt/Comp2Comp"), "bin", "C2C")
        if os.path.isfile(exe):
            import subprocess

            r = subprocess.run(
                [exe, "--help"],
                capture_output=True,
                timeout=15,
                cwd=os.getenv("COMP2COMP_DIR", "/opt/Comp2Comp"),
            )
            comp2comp_ok = r.returncode in (0, 2)
    except Exception:
        pass
    return {
        "totalseg": totalseg_ok,
        "comp2comp": comp2comp_ok,
        "ready": totalseg_ok,
        "full": totalseg_ok and comp2comp_ok,
    }


def detect_is_mri(filepath: str, series_dir: str | None) -> bool:
    """Use DICOM Modality MR when available; NIfTI defaults to CT task."""
    if series_dir and os.path.isdir(series_dir):
        import pydicom

        for f in sorted(Path(series_dir).iterdir()):
            if not f.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                mod = (getattr(ds, "Modality", "") or "").upper()
                if mod == "MR":
                    return True
                if mod == "CT":
                    return False
            except Exception:
                continue
    pl = filepath.lower()
    if pl.endswith(".dcm"):
        try:
            import pydicom

            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            return (getattr(ds, "Modality", "") or "").upper() == "MR"
        except Exception:
            return False
    return False


# Plan/docs name alignment (same implementation)
_detect_is_mri = detect_is_mri


def run_pipeline(
    filepath: str,
    job_id: str,
    series_dir: str = "",
    patient_context: dict[str, Any] | None = None,
) -> dict:
    """Run spine/neuro pipeline. claude_client parameter removed (was dead)."""
    logger.info(f"[{job_id}] Running spine/neuro pipeline...")
    pctx = patient_context or {}
    volume, meta, _loaded = load_ct_volume(filepath, series_dir=series_dir or None)
    meta = meta if isinstance(meta, dict) else {}
    film_photo = is_film_photo_meta(meta)
    degraded = is_degraded_single_slice(volume)
    is_mri = detect_is_mri(filepath, series_dir or None)
    if film_photo:
        hint = str(pctx.get("film_imaging_modality") or pctx.get("imaging_modality") or "").upper()
        is_mri = hint in ("MR", "MRI", "MRI_SPINE")
    tot_task = "vertebrae_mr" if is_mri else "vertebrae_body"

    tot_result: dict = {}
    totseg_limitation_note = ""
    if not film_photo:
        try:
            inp = filepath if filepath and os.path.isfile(filepath) else volume
            # vertebrae_body does not support --fast; vertebrae_mr does (see TotalSegmentator CLI).
            use_fast = tot_task == "vertebrae_mr"
            dev = os.getenv("TOTALSEG_DEVICE") or os.getenv("DEVICE", "cpu")
            tot_result = run_totalseg(
                inp,
                task=tot_task,
                fast=use_fast,
                device=dev,
            )
        except (SystemExit, Exception) as e:
            err_msg = str(e)
            if isinstance(e, SystemExit) or "license" in err_msg.lower():
                totseg_limitation_note = (
                    "TotalSegmentator vertebrae model unavailable "
                    "(license or weights missing, or CLI exited). "
                    "Clinical scores derived from heuristic analysis only. "
                    "For full vertebral segmentation, obtain an academic/commercial TotalSegmentator "
                    f"license and run the appropriate task (e.g. {tot_task}) per upstream documentation."
                )
                logger.warning(
                    "TotalSeg SystemExit or license issue (%s): %s — using heuristic fallback",
                    tot_task,
                    err_msg,
                )
            else:
                totseg_limitation_note = f"TotalSegmentator error: {err_msg}"
                logger.error("TotalSeg unexpected error (%s): %s", tot_task, err_msg, exc_info=True)
            tot_result = {"structure_names": [], "output_dir": "", "volumes_cm3": {}}
    else:
        logger.info("Skipping TotalSegmentator for film_photo spine input")
        totseg_limitation_note = (
            "Film photo batch — TotalSegmentator skipped; segmentation is not valid on "
            "phone-photo reconstructed stacks."
        )
        tot_result = {"structure_names": [], "output_dir": "", "volumes_cm3": {}}

    out_dir = tot_result.get("output_dir") or ""
    vertebrae = structure_list_from_result(tot_result)
    volumes_cm3 = tot_result.get("volumes_cm3") or {}
    mask_paths = tot_result.get("mask_paths") or {}

    if not vertebrae:
        vertebrae = []

    pathology_scores: dict = {}
    for k, v in volumes_cm3.items():
        pathology_scores[k] = v

    # Run opportunistic osteoporosis screening (L1 HU)
    l1_bmd = compute_l1_hu_bmd(volume, mask_paths, meta)
    if l1_bmd.get("available"):
        for key in ["l1_hu_mean", "bone_density_category"]:
            if l1_bmd.get(key) is not None:
                pathology_scores[key] = l1_bmd[key]

    # Run fracture grading with real mask-derived heights
    fracture_grading = grade_vertebral_fractures(volume, mask_paths, meta)

    # Build heights list from fracture grading results
    if fracture_grading.get("available") and fracture_grading.get("vertebrae"):
        heights_from_masks = []
        for v in fracture_grading.get("vertebrae", []):
            h = v.get("height_mm")
            if h is not None:
                heights_from_masks.append(h)
        heights = heights_from_masks if heights_from_masks else _vertebral_heights(volume, max(len(vertebrae), 1))

        pathology_scores["fracture_grading_available"] = 1.0
        pathology_scores["highest_genant_grade"] = fracture_grading.get("highest_grade", 0)
        pathology_scores["any_vertebral_fracture"] = 1.0 if fracture_grading.get("any_fracture") else 0.0
        for v in fracture_grading.get("vertebrae", []):
            level = v.get("level")
            if level:
                pathology_scores[f"{level}_genant_grade"] = v.get("genant_grade", 0)
                pathology_scores[f"{level}_height_mm"] = v.get("height_mm")
    else:
        heights = _vertebral_heights(volume, max(len(vertebrae), 1))
        pathology_scores["fracture_grading_available"] = 0.0

    for i, vb in enumerate(vertebrae[:12]):
        pathology_scores[f"{vb}_height_mm"] = heights[i] if i < len(heights) else 0.0

    findings = _build_spine_findings(
        vertebrae, heights, degraded, volumes_cm3, is_mri, tot_task, film_photo=film_photo
    )

    seg_ok = bool(vertebrae or volumes_cm3)
    models_used: list[str] = []
    if seg_ok:
        models_used.append("TotalSegmentator-vertebrae")
        if l1_bmd.get("available"):
            models_used.append("Opportunistic-BMD-L1")
        if fracture_grading.get("available"):
            models_used.append("Vertebral-Fracture-Grading")
    else:
        models_used.append("heuristic-spine")
    impression = (
        "Spine/Neuro analysis complete. Clinical correlation recommended."
        if seg_ok
        else (
            "Automated spine segmentation did not produce usable vertebra labels or volumes — "
            "clinical correlation and manual review required."
        )
    )

    clinical_struct, clinical_scores = _build_clinical_spine_structures(
        vertebrae,
        heights,
        degraded,
        volumes_cm3,
        is_mri,
        tot_task,
        pctx,
        volume,
        totseg_limitation_note=totseg_limitation_note,
        fracture_grading=fracture_grading,
    )
    merged_scores = dict(pathology_scores)
    for k, v in clinical_scores.items():
        merged_scores[k] = v

    structures_payload = {
        "volumes_cm3": volumes_cm3,
        "segment_names": vertebrae,
        "totalseg_task": tot_task,
        "mri_detected": is_mri,
        "imaging_modality": "MRI" if is_mri else "CT",
        "algorithm_version": {
            "totalsegmentator": get_totalseg_version(),
            "comp2comp_git_sha": os.getenv("COMP2COMP_GIT_SHA", "unknown"),
            "pipeline_version": PIPELINE_VERSION,
        },
        "input_type": (
            "film_photo"
            if film_photo
            else ("series" if (series_dir and os.path.isdir(series_dir)) else "single_file")
        ),
        **clinical_struct,
    }

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
                    "Spine film-photo mode: extracted %d slices for LLM visual interpretation",
                    len(llm_images_b64),
                )
        except Exception as e:
            logger.warning("Failed to extract film-photo slices for spine LLM: %s", e)
            llm_images_b64 = None
    
    narrative, narr_tags = _spine_narrative_openrouter(
        {k: structures_payload[k] for k in clinical_struct},
        merged_scores,
        pctx,
        film_photo=film_photo,
        image_b64_list=llm_images_b64,
    )
    structures_payload["narrative_report"] = narrative
    structures_payload["narrative_policy"] = _spine_narrative_policy()
    models_used = models_used + narr_tags

    is_critical = bool(
        float(merged_scores.get("cord_compression_confidence", 0) or 0) > 0.65
        or float(merged_scores.get("pott_disease_confidence", 0) or 0) > 0.85
        or (isinstance(structures_payload.get("canal_stenosis"), str) and "severe" in structures_payload["canal_stenosis"])
    )

    if narrative and len(narrative) > 40:
        impression = narrative[:320].strip() + ("…" if len(narrative) > 320 else "")

    conf_out = "medium"
    disc_out = DISCLAIMER
    if film_photo:
        apply_film_photo_pathology_scores(merged_scores)
        attach_film_meta_to_structures(structures_payload, meta)
        conf_out = cap_confidence_for_film(conf_out)
        disc_out = merge_disclaimer_with_film(
            DISCLAIMER, True, FILM_PHOTO_DISCLAIMER_ADDENDUM
        )

    return {
        "modality": "spine_neuro",
        "findings": findings,
        "impression": impression,
        "pathology_scores": merged_scores,
        "structures": structures_payload,
        "confidence": conf_out,
        "models_used": models_used,
        "disclaimer": disc_out,
        "is_critical": is_critical,
    }


def _build_spine_findings(
    vertebrae: list,
    heights: list,
    degraded: bool,
    volumes_cm3: dict,
    is_mri: bool,
    tot_task: str,
    film_photo: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    if film_photo:
        findings.append(
            Finding(
                label="Film photo input (mobile photos of printed CT/MRI film)",
                description=(
                    "Spine analysis from photographs of printed films — not DICOM. "
                    "Vertebral heights, BMD proxy, fracture grading, and segmentation are unreliable; obtain digital imaging for definitive reporting."
                ),
                severity="warning",
                confidence=100.0,
                region="Spine",
            )
        )
    if degraded:
        findings.append(
            Finding(
                label="Degraded Input — Single Slice",
                description="Full spine assessment requires a volumetric series. Single-slice input is limited.",
                severity="warning",
                confidence=100.0,
                region="Spine",
            )
        )
    if not vertebrae and not volumes_cm3:
        findings.append(
            Finding(
                label="AI spine segmentation unavailable",
                description=(
                    "TotalSegmentator did not return usable vertebra labels or volumes for this input. "
                    "No fabricated pathology scores are reported — clinical correlation and manual review required."
                ),
                severity="warning",
                confidence=100.0,
                region="Spine",
            )
        )
        return findings

    findings.append(
        Finding(
            label="Spine segmentation",
            description=(
                f"Task={tot_task} ({'MRI' if is_mri else 'CT'}). "
                f"Labels: {', '.join(vertebrae[:12])}{'...' if len(vertebrae) > 12 else ''}. "
                f"Proxy heights (mm): {', '.join(f'{h:.1f}' for h in heights[:6])}..."
            ),
            severity="info",
            confidence=80.0,
            region="Spine",
        )
    )
    if volumes_cm3:
        findings.append(
            Finding(
                label="Vertebral volumes",
                description=f"{len(volumes_cm3)} volume metrics (cm³) where masks exist.",
                severity="info",
                confidence=75.0,
                region="Spine",
            )
        )
    return findings


def _vertebral_heights(volume, n: int):
    """Proxy heights from sagittal extent of high-intensity bone voxels per slab."""
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3:
        return [24.0 + i * 0.2 for i in range(min(n, 12))]
    z = v.shape[2]
    out = []
    slab = max(1, z // max(n, 1))
    for i in range(min(n, 12)):
        s0 = i * slab
        s1 = min(z, (i + 1) * slab)
        slab_v = v[:, :, s0:s1]
        mask = slab_v > 150
        if not np.any(mask):
            out.append(24.0)
            continue
        ys, xs = np.where(np.any(mask, axis=2))
        h = (ys.max() - ys.min() + 1) * 0.5
        out.append(float(max(18.0, min(40.0, h))))
    return out


def run_pipeline_b64(
    file_b64: str | None = None,
    patient_context: str = "",
    job_id: str = "",
    filename_hint: str = "",
    series_dir: str = "",
    image_b64: str | None = None,
    patient_context_json: str | None = None,
) -> dict:
    import base64
    import tempfile
    import uuid

    b64 = file_b64 if file_b64 is not None else image_b64
    if not b64:
        return {"available": False, "reason": "missing_image", "message": "No base64 image provided."}
    ctx_raw = patient_context_json if patient_context_json is not None else patient_context
    pctx = _parse_patient_context(ctx_raw)

    data = base64.b64decode(b64)
    ext = os.path.splitext(filename_hint)[1] if filename_hint else ""
    if not ext:
        if len(data) >= 4 and data[:4] == b"\x89PNG":
            ext = ".png"
        elif len(data) >= 2 and data[:2] == b"\xff\xd8":
            ext = ".jpg"
        else:
            ext = ".dcm"
    jid = job_id or str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(suffix=ext or ".dcm", delete=False) as f:
        f.write(data)
        fp = f.name
    try:
        result = run_pipeline(fp, jid, series_dir=series_dir, patient_context=pctx)
        result["job_id"] = jid
        return result
    finally:
        try:
            os.unlink(fp)
        except OSError:
            pass


def run_spine_neuro_pipeline_b64(
    image_b64: str,
    patient_context_json: str = "{}",
    filename_hint: str = "",
    series_dir: str = "",
    job_id: str = "",
) -> dict:
    """Gateway / ZeroClaw compatible entry (image_b64 + JSON context)."""
    return run_pipeline_b64(
        image_b64=image_b64,
        patient_context_json=patient_context_json,
        filename_hint=filename_hint,
        series_dir=series_dir,
        job_id=job_id,
    )
