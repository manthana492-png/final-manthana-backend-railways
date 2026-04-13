"""Manthana — Abdominal CT Inference: TotalSeg + Comp2Comp CLI (series) + honest metrics."""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (
    os.path.normpath(os.path.join(_ROOT, "..", "..", "shared")),
    "/app/shared",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from chest_heuristics import run_chest_heuristics
from comp2comp_runner import run_comp2comp_abdominal
from disclaimer import DISCLAIMER, FILM_PHOTO_DISCLAIMER_ADDENDUM
from film_photo_reporting import (
    apply_film_photo_pathology_scores,
    attach_film_meta_to_structures,
    cap_confidence_for_film,
    is_film_photo_meta,
    merge_disclaimer_with_film,
)
from dicom_utils_helpers import count_dicoms_in_tree
from input_modality import is_mr_input
from preprocessing.ct_loader import is_degraded_single_slice, load_ct_volume
from schemas import Finding
from sybil_runner import run_sybil
from totalseg_runner import (
    estimate_aortic_diameter_mm,
    get_totalseg_version,
    run_totalseg,
    structure_list_from_result,
)

import config as abdominal_config

logger = logging.getLogger("manthana.abdominal_ct")

PIPELINE_VERSION = "manthana-ct-v2"

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_CT_SYSTEM = _PROMPT_DIR / "ct_system.md"

# Backwards compatibility alias
_count_dicoms_in_tree = count_dicoms_in_tree


def _is_raster_path(path: str, filename_hint: str = "") -> bool:
    for p in (filename_hint or "", path or ""):
        pl = p.lower()
        if pl.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
            return True
    return False


def _stage_single_dicom_file(filepath: str, job_id: str) -> tuple[str, list[str]]:
    """Copy lone .dcm into temp dir so TotalSegmentator sees a series folder."""
    cleanup: list[str] = []
    if not filepath or not os.path.isfile(filepath):
        return filepath, cleanup
    low = filepath.lower()
    if not (low.endswith(".dcm") or low.endswith(".dic")):
        return filepath, cleanup
    import tempfile

    d = tempfile.mkdtemp(prefix=f"ct_stage_{job_id}_")
    cleanup.append(d)
    shutil.copy2(filepath, os.path.join(d, os.path.basename(filepath) or "instance.dcm"))
    return d, cleanup


def _read_ct_system_prompt() -> str:
    try:
        if _CT_SYSTEM.is_file():
            return _CT_SYSTEM.read_text(encoding="utf-8")[:16000]
    except OSError:
        pass
    return (
        "You are a senior radiologist. Summarise abdominal CT findings conservatively from JSON only. "
        "India: HBV/HCV, NAFLD context; note TotalSegmentator training cohort limitations."
    )


def _abdominal_narrative_policy() -> str:
    """off = disabled; anything else = OpenRouter (SSOT: config/cloud_inference.yaml)."""
    v = (os.environ.get("CT_ABDOMINAL_NARRATIVE_POLICY", "openrouter") or "openrouter").strip().lower()
    if v in ("off", "none", "disabled", "0"):
        return "off"
    return "openrouter"


def _call_ct_narrative(
    *,
    pathology_scores: dict,
    impression: str,
    findings: list,
    patient_context: dict | None,
    film_photo: bool = False,
    image_b64_list: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Controlled by CT_ABDOMINAL_NARRATIVE_POLICY (off | openrouter).
    Never raises; returns empty narrative if all fail or policy is off.
    
    For film_photo mode: can send multiple representative slices for visual interpretation.
    """
    policy = _abdominal_narrative_policy()
    if policy == "off":
        return "", ["CT-narrative-disabled"]

    tags: list[str] = []
    system = _read_ct_system_prompt()
    if film_photo:
        system = FILM_PHOTO_NARRATIVE_PREFIX + system
    scores_json = json.dumps(pathology_scores, indent=2)[:12000]
    patient_json = json.dumps(patient_context or {}, indent=2)[:4000]
    findings_json = json.dumps(findings, indent=2)[:8000]

    ctx = patient_context or {}
    contrast = ctx.get("contrast_phase")
    contrast_line = (
        f"CONTRAST_PHASE: {contrast}\n"
        "(NCCT vs CECT changes vascular and solid-organ interpretation — stay consistent.)\n\n"
        if contrast
        else ""
    )

    india_block = (
        "India-specific framing (population context; do not contradict JSON):\n"
        "- Liver: HBV/HCV remain important; HBsAg carrier rate ~3.7% in national surveys (population context only). "
        "NAFLD rising in urban India; alcohol-related disease common in men.\n"
        "- Pancreas: tropical pancreatitis endemic in Kerala/coastal India — distinct from Western alcoholic pancreatitis; "
        "TPIAT context when relevant.\n"
        "- Kidneys: renal calculi highly prevalent in the \"stone belt\" (Rajasthan, Maharashtra, Gujarat).\n"
        "- Aorta: aneurysm threshold commonly ~3 cm for surveillance; **aortic dissection → EMERGENCY**.\n"
        "EMERGENCY flags (lead first if findings/scores support; do not fabricate):\n"
        "- Free intraperitoneal air → perforation → EMERGENCY surgery.\n"
        "- Bowel obstruction / closed-loop → surgical consult URGENT.\n"
        "- Aortic dissection → ICU EMERGENCY.\n"
        "- Rupture / hemodynamic compromise signs → EMERGENCY.\n"
    )

    user_text = (
        contrast_line
        + f"IMPRESSION (pipeline):\n{impression}\n\n"
        f"STRUCTURED_FINDINGS:\n{findings_json}\n\n"
        f"PATHOLOGY_SCORES (numeric):\n{scores_json}\n\n"
        f"PATIENT_CONTEXT:\n{patient_json}\n\n"
        f"{india_block}\n"
        "Write a concise narrative: EMERGENCY block first if warranted by findings/scores, "
        "then summary, then India-context clinical correlates. Do not invent measurements or lesions."
    )

    try:
        from llm_router import llm_router

        # For film-photo mode with multi-image vision
        has_vision_images = film_photo and image_b64_list and len(image_b64_list) > 0
        
        if has_vision_images:
            logger.info("Abdominal CT narrative: using multi-image vision with %d film-photo slices", len(image_b64_list))
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
                max_tokens=1600,
            )
        txt = (out.get("content") or "").strip()
        if txt:
            tags.append("OpenRouter-narrative-CT")
            return txt, tags
    except Exception as e:
        logger.warning("Abdominal CT OpenRouter narrative failed: %s", e)
    return "", tags


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _clinical_input_type(
    *,
    work_fp: str,
    series_available: bool,
    filename_hint: str,
    dicom_n: int,
) -> str:
    """structures.input_type: dicom_series | single_slice | png_jpeg"""
    if _is_raster_path(work_fp, filename_hint):
        return "png_jpeg"
    if series_available or os.path.isdir(work_fp):
        return "dicom_series" if dicom_n >= 2 else "single_slice"
    low = work_fp.lower()
    if low.endswith((".nii", ".nii.gz")) or ".nii.gz" in low:
        return "dicom_series"
    if low.endswith((".dcm", ".dic")):
        return "single_slice"
    return "dicom_series"


def _plan_totalseg(
    patient_context: dict | None,
    dicom_slices_found: int,
    *,
    is_raster_upload: bool,
) -> tuple[bool, bool, str, str]:
    """
    Returns (run_totalseg, fast_flag, segmentation_quality, totalseg_model_used).
    segmentation_quality: full_3d | degraded | visual_only
    """
    ctx = patient_context or {}
    upload_type = str(ctx.get("upload_type") or "").lower()

    if is_raster_upload or upload_type == "image_files":
        return False, False, "visual_only", "none"

    declared = _safe_int(ctx.get("declared_file_count"))
    n = dicom_slices_found if dicom_slices_found > 0 else declared

    override = str(ctx.get("totalseg_model") or "").lower()
    if override in ("none", "skip", "off"):
        return False, False, "visual_only", "none"

    if n < 30:
        return False, False, "visual_only", "none"

    if n >= 80:
        fast = False
        qual = "full_3d"
        used = "full"
    else:
        fast = True
        qual = "degraded"
        used = "fast"

    if override == "fast":
        fast, qual, used = True, "degraded", "fast"
    elif override == "full":
        fast, qual, used = False, "full_3d", "full"

    return True, fast, qual, used


def is_loaded() -> dict:
    """Component availability for /health (totalseg + optional Comp2Comp binary + sybil)."""
    totalseg_ok = False
    comp2comp_ok = False
    sybil_ok = False
    try:
        import totalsegmentator  # noqa: F401

        totalseg_ok = True
    except ImportError:
        pass
    try:
        import subprocess

        exe = os.path.join(os.getenv("COMP2COMP_DIR", "/opt/Comp2Comp"), "bin", "C2C")
        if os.path.isfile(exe):
            r = subprocess.run(
                [exe, "--help"],
                capture_output=True,
                timeout=15,
                cwd=os.getenv("COMP2COMP_DIR", "/opt/Comp2Comp"),
            )
            comp2comp_ok = r.returncode in (0, 2)  # some CLIs return 2 for help
        else:
            comp2comp_ok = False
    except Exception:
        comp2comp_ok = False
    try:
        import sybil  # noqa: F401

        sybil_ok = True
    except ImportError:
        pass
    return {"totalseg": totalseg_ok, "comp2comp": comp2comp_ok, "sybil": sybil_ok}


def _float_pathology_scores(scores: dict) -> dict:
    """E2E contract: numeric pathology_scores only (floats); omit None / non-scalars."""
    out: dict = {}
    for k, v in scores.items():
        if v is None:
            continue
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def run_pipeline(
    filepath: str,
    job_id: str,
    series_dir: str = "",
    source_modality: str = "",
    patient_context: dict | None = None,
    http_upload_filename: str = "",
    dicom_slices_found: int | None = None,
    dicom_declared_mismatch: bool = False,
    **kwargs: Any,
) -> dict:
    _ = kwargs
    cleanup_infer: list[str] = []
    try:
        logger.info(f"[{job_id}] Running abdominal CT pipeline (series_dir={bool(series_dir)})...")
        ext_series = bool(series_dir and os.path.isdir(series_dir))

        work_fp = filepath
        if not ext_series:
            staged, cdir = _stage_single_dicom_file(filepath, job_id)
            cleanup_infer.extend(cdir)
            work_fp = staged

        series_available = ext_series or (os.path.isdir(work_fp) and _count_dicoms_in_tree(work_fp) >= 1)

        volume, vol_meta, loaded_from_path = load_ct_volume(work_fp, series_dir=series_dir or None)
        vol_meta = vol_meta if isinstance(vol_meta, dict) else {}
        film_photo = is_film_photo_meta(vol_meta)
        if ext_series:
            series_available = True
        elif loaded_from_path and os.path.isdir(work_fp):
            series_available = True

        degraded = is_degraded_single_slice(volume)

        if ext_series and os.path.isdir(series_dir):
            tot_input: Any = series_dir
            c2c_dir: str | None = series_dir
        elif os.path.isdir(work_fp):
            tot_input = work_fp
            c2c_dir = work_fp
        elif os.path.isfile(work_fp):
            tot_input = work_fp
            c2c_dir = None
        else:
            tot_input = volume
            c2c_dir = None

        dicom_n = (
            dicom_slices_found
            if dicom_slices_found is not None
            else (_count_dicoms_in_tree(work_fp) if os.path.isdir(work_fp) else 0)
        )
        is_raster = _is_raster_path(work_fp, http_upload_filename)
        run_ts, fast_flag, seg_qual, ts_used = _plan_totalseg(
            patient_context,
            dicom_n,
            is_raster_upload=is_raster,
        )
        if seg_qual == "visual_only":
            c2c_dir = None

        tot_result: dict = {}
        if run_ts:
            try:
                tot_result = run_totalseg(
                    tot_input,
                    task="total",
                    fast=fast_flag,
                    device=os.getenv("TOTALSEG_DEVICE", "gpu"),
                )
            except Exception as e:
                logger.warning("TotalSegmentator failed: %s", e)
                tot_result = {"structure_names": [], "output_dir": "", "volumes_cm3": {}}
        else:
            tot_result = {"structure_names": [], "output_dir": "", "volumes_cm3": {}}

        out_dir = tot_result.get("output_dir") or ""
        segments = structure_list_from_result(tot_result)
        volumes_cm3 = tot_result.get("volumes_cm3") or {}

        if not segments and not film_photo:
            segments = [
                "liver",
                "spleen",
                "right_kidney",
                "left_kidney",
                "pancreas",
                "aorta",
                "ivc",
                "gallbladder",
                "stomach",
                "bowel",
                "vertebrae",
            ]

        aorta_metrics = estimate_aortic_diameter_mm(out_dir) if out_dir else {}
        comp2comp_results = run_comp2comp_abdominal(
            work_fp if os.path.isfile(work_fp) else (work_fp or ""),
            volume=volume,
            series_dir=c2c_dir,
        )

        scores: dict = {}
        for k, v in volumes_cm3.items():
            scores[k] = v

        scores["max_aorta_diameter_mm"] = aorta_metrics.get("max_aorta_diameter_mm")
        if aorta_metrics.get("aaa_detected") is not None:
            scores["aaa_flag"] = float(bool(aorta_metrics.get("aaa_detected")))
        if aorta_metrics.get("aaa_risk_flag") is not None:
            scores["aaa_risk_flag"] = float(bool(aorta_metrics.get("aaa_risk_flag")))

        scores["bmd_score"] = comp2comp_results.get("bmd_score")
        scores["low_bmd_flag"] = float(bool(comp2comp_results.get("low_bmd_flag", False)))
        if comp2comp_results.get("t_score_estimate") is not None:
            scores["t_score_estimate"] = comp2comp_results.get("t_score_estimate")
        if comp2comp_results.get("muscle_area_cm2") is not None:
            scores["muscle_area_cm2"] = comp2comp_results.get("muscle_area_cm2")
        if comp2comp_results.get("visceral_fat_cm2") is not None:
            scores["visceral_fat_cm2"] = comp2comp_results.get("visceral_fat_cm2")

        ctx_pc = patient_context or {}

        # Chest CT routing: Sybil lung cancer risk + chest heuristics
        ct_region = str(ctx_pc.get("ct_region") or "").lower()
        is_chest_ct = ct_region in ("chest", "thorax", "lung")
        sybil_results: dict = {}
        chest_heuristic_results: dict = {}

        if is_chest_ct and abdominal_config.SYBIL_ENABLED:
            dicom_paths_for_sybil = []
            if ext_series and os.path.isdir(series_dir):
                from dicom_utils_helpers import find_dicom_files
                dicom_paths_for_sybil = find_dicom_files(series_dir)
            elif os.path.isdir(work_fp):
                from dicom_utils_helpers import find_dicom_files
                dicom_paths_for_sybil = find_dicom_files(work_fp)
            
            if dicom_paths_for_sybil:
                sybil_results = run_sybil(dicom_paths_for_sybil)
                if sybil_results.get("available"):
                    for key in ["risk_1yr", "risk_2yr", "risk_3yr", "risk_4yr", "risk_5yr", "risk_6yr"]:
                        if sybil_results.get(key) is not None:
                            scores[f"sybil_{key}"] = sybil_results[key]
                    scores["sybil_risk_category"] = sybil_results.get("risk_category")
                    scores["sybil_recommend_ldct"] = float(sybil_results.get("recommend_followup_ldct", False))

        # Chest heuristics (TB, NAFLD, tropical pancreatitis) for chest or abdomen CT
        if abdominal_config.TB_HEURISTIC_ENABLED and out_dir:
            mask_paths = tot_result.get("mask_paths") or {}
            chest_heuristic_results = run_chest_heuristics(volume, mask_paths, region=ct_region if ct_region else "abdomen")
            # Flatten TB heuristic into scores
            tb_heur = chest_heuristic_results.get("tb_heuristic", {})
            if tb_heur.get("tb_score") is not None:
                scores["tb_heuristic_score"] = tb_heur["tb_score"]
                scores["tb_suspect"] = float(tb_heur.get("tb_suspect", False))
            # NAFLD
            nafld = chest_heuristic_results.get("nafld_heuristic", {})
            if nafld.get("nafld_grade"):
                scores["nafld_grade"] = nafld["nafld_grade"]
            if nafld.get("liver_mean_hu") is not None:
                scores["liver_mean_hu"] = nafld["liver_mean_hu"]
            if nafld.get("spleen_mean_hu") is not None:
                scores["spleen_mean_hu"] = nafld["spleen_mean_hu"]
            # Tropical pancreatitis
            tp = chest_heuristic_results.get("tropical_pancreatitis_heuristic", {})
            if tp.get("tropical_pancreatitis_suspect") is not None:
                scores["tropical_pancreatitis_suspect"] = float(tp["tropical_pancreatitis_suspect"])
            if tp.get("pancreatic_calc_count") is not None:
                scores["pancreatic_calc_count"] = tp["pancreatic_calc_count"]

        if run_ts:
            models_used = ["TotalSegmentator-v2", "TotalSegmentator-AAQ-proxy"]
        else:
            models_used = ["Manthana-CT-heuristic"]
        c2c = comp2comp_results.get("c2c_series") or {}
        if isinstance(c2c, dict) and c2c.get("source") == "comp2comp_series":
            models_used.extend(
                [
                    "Comp2Comp-spine",
                    "Comp2Comp-liver_spleen_pancreas",
                    "Comp2Comp-spine_muscle_adipose_tissue",
                ]
            )
        # Add Sybil lung cancer risk model tag if chest CT and sybil ran
        if is_chest_ct and sybil_results.get("available"):
            models_used.append("Sybil-Lung-Cancer-Risk")
        models_used = list(dict.fromkeys(models_used))

        clinical_input_type = _clinical_input_type(
            work_fp=work_fp,
            series_available=series_available,
            filename_hint=http_upload_filename,
            dicom_n=dicom_n,
        )

        contrast_phase = str(ctx_pc.get("contrast_phase") or "unknown")

        organ_structures_measured = 0
        if run_ts and isinstance(volumes_cm3, dict):
            organ_structures_measured = len(
                [
                    k
                    for k, v in volumes_cm3.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0
                ]
            )

        structures_payload: dict = {
            "volumes_cm3": volumes_cm3,
            "segment_names": segments,
            "algorithm_version": {
                "totalsegmentator": get_totalseg_version(),
                "comp2comp_git_sha": os.getenv("COMP2COMP_GIT_SHA", "unknown"),
                "pipeline_version": PIPELINE_VERSION,
            },
            "input_type": clinical_input_type,
            "segmentation_quality": seg_qual,
            "dicom_slices_found": dicom_n,
            "contrast_phase": contrast_phase,
            "totalseg_model_used": ts_used,
            "totalseg_fast_mode": bool(fast_flag) if run_ts else None,
            "dicom_declared_mismatch": bool(dicom_declared_mismatch),
            "organ_structures_measured": organ_structures_measured,
            "routing_series_dir": ext_series,
            "narrative_report": "",
        }

        mr_study = is_mr_input(source_modality, series_dir or None)
        skipped_protocol = (not run_ts) and ts_used == "none" and (dicom_n > 0 or is_raster)
        findings = _build_findings(
            segments,
            comp2comp_results,
            aorta_metrics,
            degraded,
            series_available,
            mr_study,
            totalseg_skipped_protocol=skipped_protocol,
            sybil_results=sybil_results,
            chest_heuristic_results=chest_heuristic_results,
            is_chest_ct=is_chest_ct,
            film_photo=film_photo,
        )
        findings_out = [f.model_dump() if isinstance(f, Finding) else f for f in findings]

        impression = "Abdominal CT analysis complete. Clinical correlation recommended."
        scores_out = _float_pathology_scores(scores)

        # Extract representative slices for LLM vision in film-photo mode
        llm_images_b64: list[str] | None = None
        if film_photo and vol_meta:
            try:
                from film_photo_reporting import extract_film_photo_images_for_llm
                llm_images_b64 = extract_film_photo_images_for_llm(
                    vol_meta,
                    max_images=10,
                    min_quality_threshold=30.0,
                )
                if llm_images_b64:
                    logger.info(
                        "Abdominal CT film-photo mode: extracted %d slices for LLM visual interpretation",
                        len(llm_images_b64),
                    )
            except Exception as e:
                logger.warning("Failed to extract film-photo slices for abdominal CT LLM: %s", e)
                llm_images_b64 = None
        
        narrative, ct_narr_tags = _call_ct_narrative(
            pathology_scores=scores_out,
            impression=impression,
            findings=findings_out,
            patient_context=patient_context,
            film_photo=film_photo,
            image_b64_list=llm_images_b64,
        )
        structures_payload["narrative_policy"] = _abdominal_narrative_policy()
        for t in ct_narr_tags:
            if t not in models_used:
                models_used.append(t)
        if narrative:
            structures_payload["narrative_report"] = narrative

        conf_out = "medium"
        disc_out = DISCLAIMER
        if film_photo:
            apply_film_photo_pathology_scores(scores_out)
            attach_film_meta_to_structures(structures_payload, vol_meta)
            conf_out = cap_confidence_for_film(conf_out)
            disc_out = merge_disclaimer_with_film(
                DISCLAIMER, True, FILM_PHOTO_DISCLAIMER_ADDENDUM
            )

        return {
            "modality": "abdominal_ct",
            "findings": findings_out,
            "impression": impression,
            "pathology_scores": scores_out,
            "structures": structures_payload,
            "confidence": conf_out,
            "models_used": models_used,
            "disclaimer": disc_out,
        }
    finally:
        for d in cleanup_infer:
            shutil.rmtree(d, ignore_errors=True)


def _build_findings(
    segments: list,
    comp2comp: dict,
    aorta: dict,
    degraded: bool,
    series_available: bool,
    mr_study: bool = False,
    totalseg_skipped_protocol: bool = False,
    sybil_results: dict | None = None,
    chest_heuristic_results: dict | None = None,
    is_chest_ct: bool = False,
    film_photo: bool = False,
) -> list:
    findings: list = []

    if film_photo:
        findings.append(
            Finding(
                label="Film photo input (mobile photos of printed CT/MRI film)",
                description=(
                    "This analysis used mobile phone photographs of printed imaging films, not original DICOM. "
                    "Volumetric measurements, organ segmentation, Sybil risk, and quantitative scores are "
                    "unreliable. For definitive care, obtain digital DICOM from the imaging centre."
                ),
                severity="warning",
                confidence=100.0,
                region="General",
            )
        )

    if totalseg_skipped_protocol:
        findings.append(
            Finding(
                label="Volumetric segmentation omitted (protocol)",
                description=(
                    "TotalSegmentator was not run: fewer than 30 DICOM slices, image upload, or explicit skip. "
                    "Organ volumes are not measured — heuristic / visual assessment only. "
                    "Upload 30+ slices (80+ recommended for full volumetrics) for 3D segmentation."
                ),
                severity="warning",
                confidence=98.0,
                region="Abdomen",
            )
        )

    if mr_study:
        findings.append(
            Finding(
                label="MRI processed by CT-optimised pipeline",
                description=(
                    "This abdominal study was routed as CT-optimised TotalSegmentator / Comp2Comp. "
                    "If the source is MRI, results may be inaccurate — correlate with a dedicated MRI protocol."
                ),
                severity="warning",
                confidence=100.0,
                region="Abdomen",
            )
        )

    if degraded:
        findings.append(
            Finding(
                label="Degraded Input — Single Slice",
                description=(
                    "Single-slice or thin input detected. Organ segmentation and measurements "
                    "are most reliable on a full CT volume. Upload a complete DICOM series when possible."
                ),
                severity="warning",
                confidence=100.0,
                region="Abdomen",
            )
        )

    if not series_available:
        findings.append(
            Finding(
                label="Comp2Comp series metrics unavailable",
                description=(
                    "Bone density and body-composition pipelines need a full DICOM series directory. "
                    "Manual single-file upload uses TotalSegmentator and HU fallbacks only."
                ),
                severity="info",
                confidence=95.0,
                region="Abdomen",
            )
        )

    aorta_mm = aorta.get("max_aorta_diameter_mm")
    if aorta_mm is not None and aorta_mm > 0:
        if aorta.get("aaa_risk_flag"):
            sev = "critical"
        elif aorta.get("aaa_detected"):
            sev = "warning"
        else:
            sev = "clear"
        findings.append(
            Finding(
                label="Aorta measurement (TotalSeg mask)",
                description=f"Estimated max aortic diameter {aorta_mm:.1f} mm (mask-based; not FDA-cleared device output).",
                severity=sev,
                confidence=80.0,
                region="Aorta",
            )
        )

    bmd = comp2comp.get("bmd_score")
    low_bmd = comp2comp.get("low_bmd_flag", False)
    t_score = comp2comp.get("t_score_estimate")
    if low_bmd:
        findings.append(
            Finding(
                label="Low Bone Mineral Density (proxy)",
                description=f"BMD-related score {bmd}, T-score estimate {t_score}. Consider DXA confirmation.",
                severity="warning",
                confidence=75.0,
                region="Spine",
            )
        )
    elif bmd is not None and bmd > 0:
        findings.append(
            Finding(
                label="Bone density assessment",
                description=f"BMD-related score {bmd:.1f}, T-score estimate {t_score}.",
                severity="clear",
                confidence=70.0,
                region="Spine",
            )
        )

    if segments:
        findings.append(
            Finding(
                label="Organ segmentation",
                description=f"{len(segments)} structures listed; volumes in pathology_scores where available.",
                severity="info",
                confidence=90.0,
                region="Abdomen",
            )
        )

    # Chest CT specific findings (Sybil, TB heuristics)
    if is_chest_ct and sybil_results:
        if sybil_results.get("available"):
            risk_cat = sybil_results.get("risk_category", "unknown")
            risk_6yr = sybil_results.get("risk_6yr", 0.0)
            sev = "clear" if risk_cat == "low" else ("warning" if risk_cat == "moderate" else "critical")
            findings.append(
                Finding(
                    label=f"Lung cancer risk (Sybil): {risk_cat}",
                    description=(
                        f"6-year lung cancer risk: {risk_6yr:.3f} "
                        f"(1-year: {sybil_results.get('risk_1yr', 0.0):.3f}). "
                        f"Follow-up LDCT recommended: {sybil_results.get('recommend_followup_ldct', False)}"
                    ),
                    severity=sev,
                    confidence=85.0,
                    region="Chest",
                )
            )
        elif sybil_results.get("reason") == "sybil_not_installed":
            findings.append(
                Finding(
                    label="Sybil lung cancer risk: unavailable",
                    description="Sybil package not installed. Install 'sybil>=1.6.0' for lung cancer risk prediction.",
                    severity="info",
                    confidence=100.0,
                    region="Chest",
                )
            )

    # TB heuristic findings
    if chest_heuristic_results:
        tb_heur = chest_heuristic_results.get("tb_heuristic", {})
        if tb_heur.get("tb_suspect"):
            findings.append(
                Finding(
                    label="TB pattern suspected (heuristic)",
                    description="; ".join(tb_heur.get("findings", ["Apical consolidation or calcified granulomas detected"])),
                    severity="warning",
                    confidence=60.0,
                    region="Chest",
                )
            )
        
        # NAFLD
        nafld = chest_heuristic_results.get("nafld_heuristic", {})
        if nafld.get("nafld_grade") and nafld["nafld_grade"] != "none":
            findings.append(
                Finding(
                    label=f"NAFLD grade: {nafld['nafld_grade']}",
                    description=f"Liver HU={nafld.get('liver_mean_hu')}, Spleen HU={nafld.get('spleen_mean_hu')}, ratio={nafld.get('liver_spleen_ratio')}",
                    severity="warning" if nafld["nafld_grade"] in ("moderate", "severe") else "info",
                    confidence=70.0,
                    region="Abdomen",
                )
            )
        
        # Tropical pancreatitis
        tp = chest_heuristic_results.get("tropical_pancreatitis_heuristic", {})
        if tp.get("tropical_pancreatitis_suspect"):
            findings.append(
                Finding(
                    label="Tropical pancreatitis suspected",
                    description=f"Pancreatic calcifications: {tp.get('pancreatic_calc_count', 0)}. Correlate with clinical history (Kerala/coastal India endemic).",
                    severity="warning",
                    confidence=55.0,
                    region="Abdomen",
                )
            )

    if not findings:
        findings.append(
            Finding(
                label="No significant abnormality",
                description="Abdominal CT analysis complete. No significant abnormality flagged automatically.",
                severity="clear",
                confidence=80.0,
                region="Abdomen",
            )
        )

    return findings


def run_pipeline_b64(
    file_b64: str,
    patient_context: str = "",
    job_id: str = "",
    filename_hint: str = "",
    series_dir: str = "",
) -> dict:
    import base64
    import tempfile
    import uuid

    data = base64.b64decode(file_b64)
    ext = os.path.splitext(filename_hint)[1] if filename_hint else ".dcm"
    jid = job_id or str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(suffix=ext or ".dcm", delete=False) as f:
        f.write(data)
        fp = f.name
    ctx_dict: dict | None = None
    raw_pc = (patient_context or "").strip()
    if raw_pc:
        try:
            p = json.loads(raw_pc)
            if isinstance(p, dict):
                ctx_dict = p
        except json.JSONDecodeError:
            ctx_dict = None
    try:
        return run_pipeline(
            fp,
            jid,
            series_dir=series_dir,
            http_upload_filename=filename_hint or "",
            patient_context=ctx_dict,
        )
    finally:
        try:
            os.unlink(fp)
        except OSError:
            pass
