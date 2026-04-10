"""Manthana — Cardiac CT Inference: TotalSeg heartchambers + mask-based aorta metrics + CAC scoring."""

import json
import logging
import os
import sys

sys.path.insert(0, "/app/shared")

from calcium_scoring import run_calcium_scoring
from cardiac_heuristics import run_cardiac_heuristics
from disclaimer import DISCLAIMER, FILM_PHOTO_DISCLAIMER_ADDENDUM
from film_photo_reporting import (
    FILM_PHOTO_NARRATIVE_PREFIX,
    apply_film_photo_pathology_scores,
    attach_film_meta_to_structures,
    cap_confidence_for_film,
    is_film_photo_meta,
    merge_disclaimer_with_film,
)
from input_modality import is_mr_input
from preprocessing.ct_loader import is_degraded_single_slice, load_ct_volume
from schemas import Finding
from totalseg_runner import (
    estimate_aortic_diameter_mm,
    get_totalseg_version,
    run_totalseg,
    structure_list_from_result,
)

logger = logging.getLogger("manthana.cardiac_ct")
PIPELINE_VERSION = "manthana-ct-v2"


def _cardiac_narrative_policy() -> str:
    """
    CT_CARDIAC_NARRATIVE_POLICY:
      - openrouter (default): OpenRouter narrative via config/cloud_inference.yaml
      - off | none | disabled | 0: disabled
    """
    v = (os.environ.get("CT_CARDIAC_NARRATIVE_POLICY", "openrouter") or "openrouter").strip().lower()
    if v in ("off", "none", "disabled", "0"):
        return "off"
    return "openrouter"


def _call_cardiac_ct_narrative(
    *,
    impression: str,
    findings: list,
    pathology_scores: dict,
    patient_context: dict | None,
    film_photo: bool = False,
    image_b64_list: list[str] | None = None,
) -> tuple[str, list[str]]:
    policy = _cardiac_narrative_policy()
    tags: list[str] = []
    if policy in ("off", "none", "disabled", "0"):
        return "", tags

    system = (
        "You are a cardiac imaging specialist writing a concise CT narrative from structured outputs only. "
        "Do not invent measurements, stenosis grades, or calcium scores not present in the JSON."
    )
    if film_photo:
        system = FILM_PHOTO_NARRATIVE_PREFIX + system
    findings_json = json.dumps(findings, indent=2)[:8000]
    scores_json = json.dumps(pathology_scores, indent=2)[:8000]
    patient_json = json.dumps(patient_context or {}, indent=2)[:4000]
    user_text = (
        f"IMPRESSION (pipeline):\n{impression}\n\n"
        f"FINDINGS:\n{findings_json}\n\n"
        f"PATHOLOGY_SCORES:\n{scores_json}\n\n"
        f"PATIENT_CONTEXT:\n{patient_json}"
    )

    try:
        from llm_router import llm_router

        # For film-photo mode with multi-image vision
        has_vision_images = film_photo and image_b64_list and len(image_b64_list) > 0
        
        if has_vision_images:
            logger.info("Cardiac CT narrative: using multi-image vision with %d film-photo slices", len(image_b64_list))
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
            tags.append("OpenRouter-narrative-Cardiac")
            return txt, tags
    except Exception as e:
        logger.warning("Cardiac CT OpenRouter narrative failed: %s", e)
    return "", tags


def is_loaded() -> dict:
    totalseg_ok = False
    try:
        import totalsegmentator  # noqa: F401

        totalseg_ok = True
    except ImportError:
        pass
    # Note: Comp2Comp not used in this pipeline; removed from health check
    return {"totalseg": totalseg_ok, "comp2comp": None}


def run_pipeline(
    filepath: str,
    job_id: str,
    series_dir: str = "",
    source_modality: str = "",
    patient_context: dict | None = None,
) -> dict:
    logger.info(f"[{job_id}] Running cardiac CT pipeline...")
    volume, meta, _loaded = load_ct_volume(filepath, series_dir=series_dir or None)
    meta = meta if isinstance(meta, dict) else {}
    film_photo = is_film_photo_meta(meta)
    degraded = is_degraded_single_slice(volume)

    tot_result: dict = {}
    if not film_photo:
        try:
            inp = filepath if filepath and os.path.isfile(filepath) else volume
            tot_result = run_totalseg(
                inp,
                task="heartchambers",
                fast=True,
                device=os.getenv("TOTALSEG_DEVICE", "gpu"),
            )
        except Exception as e:
            logger.warning("TotalSegmentator cardiac task failed: %s", e)
            tot_result = {"structure_names": [], "output_dir": "", "volumes_cm3": {}}
    else:
        logger.info("Skipping TotalSegmentator for film_photo cardiac input")
        tot_result = {"structure_names": [], "output_dir": "", "volumes_cm3": {}}

    out_dir = tot_result.get("output_dir") or ""
    segments = structure_list_from_result(tot_result)
    volumes_cm3 = tot_result.get("volumes_cm3") or {}

    if not segments and not film_photo:
        segments = [
            "left_ventricle",
            "right_ventricle",
            "left_atrium",
            "right_atrium",
            "aorta",
            "pulmonary_artery",
            "pericardium",
        ]

    aorta_metrics = estimate_aortic_diameter_mm(out_dir) if out_dir else {}

    pathology_scores: dict = {}
    for k, v in volumes_cm3.items():
        pathology_scores[k] = v
    if aorta_metrics.get("max_aorta_diameter_mm") is not None:
        pathology_scores["max_aorta_diameter_mm"] = aorta_metrics.get("max_aorta_diameter_mm")
    if aorta_metrics.get("aaa_detected") is not None:
        pathology_scores["aaa_flag"] = float(bool(aorta_metrics.get("aaa_detected")))
    if aorta_metrics.get("aaa_risk_flag") is not None:
        pathology_scores["aaa_risk_flag"] = float(bool(aorta_metrics.get("aaa_risk_flag")))

    # Run calcium scoring and cardiac heuristics if masks available
    mask_paths = tot_result.get("mask_paths") or {}
    cac_results = run_calcium_scoring(volume, mask_paths, meta)
    if cac_results.get("cac_available"):
        pathology_scores["cac_agatston_score"] = cac_results.get("cac_agatston_score")
        pathology_scores["cac_risk_category"] = cac_results.get("cac_risk_category")
        pathology_scores["cac_volume_mm3"] = cac_results.get("cac_volume_mm3")

    cardiac_heur = run_cardiac_heuristics(volume, mask_paths, meta)
    peri = cardiac_heur.get("pericardial", {})
    if peri.get("available"):
        pathology_scores["pericardial_effusion_suspected"] = float(peri.get("pericardial_effusion_suspected", False))
        pathology_scores["pericardial_effusion_volume_ml"] = peri.get("pericardial_effusion_volume_ml")
    cardio = cardiac_heur.get("cardiomegaly", {})
    if cardio.get("available"):
        pathology_scores["cardiomegaly_index"] = cardio.get("cardiomegaly_index")
        pathology_scores["cardiomegaly_suspected"] = float(cardio.get("cardiomegaly_suspected", False))

    mr_study = is_mr_input(source_modality, series_dir or None)
    findings = _build_cardiac_findings(
        segments, aorta_metrics, cac_results, cardiac_heur, degraded, mr_study, film_photo=film_photo
    )
    findings_out = [f.model_dump() if isinstance(f, Finding) else f for f in findings]

    impression = "Cardiac CT analysis complete. Clinical correlation recommended."
    
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
                    "Cardiac CT film-photo mode: extracted %d slices for LLM visual interpretation",
                    len(llm_images_b64),
                )
        except Exception as e:
            logger.warning("Failed to extract film-photo slices for cardiac CT LLM: %s", e)
            llm_images_b64 = None
    
    narrative, narr_tags = _call_cardiac_ct_narrative(
        impression=impression,
        findings=findings_out,
        pathology_scores=pathology_scores,
        patient_context=patient_context,
        film_photo=film_photo,
        image_b64_list=llm_images_b64,
    )

    structures_payload = {
        "volumes_cm3": volumes_cm3,
        "segment_names": segments,
        "algorithm_version": {
            "totalsegmentator": get_totalseg_version(),
            "comp2comp_git_sha": os.getenv("COMP2COMP_GIT_SHA", "unknown"),
            "pipeline_version": PIPELINE_VERSION,
        },
        "input_type": "series" if (series_dir and os.path.isdir(series_dir)) else "single_file",
        "narrative_report": narrative,
        "narrative_policy": _cardiac_narrative_policy(),
    }
    if film_photo:
        structures_payload["input_type"] = "film_photo"
    if narrative and len(narrative) > 40:
        impression = narrative[:320].strip() + ("…" if len(narrative) > 320 else "")

    models_used: list[str] = []
    if not film_photo:
        models_used.extend(["TotalSegmentator-heartchambers", "TotalSegmentator-AAQ-proxy"])
    else:
        models_used.append("Film-photo-stack")
    if cac_results.get("cac_available"):
        models_used.append("CAC-Agatston-HU")
    if peri.get("available") or cardio.get("available"):
        models_used.append("Cardiac-Heuristics")
    models_used.extend(narr_tags)

    conf_out = "medium"
    disc_out = DISCLAIMER
    if film_photo:
        apply_film_photo_pathology_scores(pathology_scores)
        attach_film_meta_to_structures(structures_payload, meta)
        conf_out = cap_confidence_for_film(conf_out)
        disc_out = merge_disclaimer_with_film(
            DISCLAIMER, True, FILM_PHOTO_DISCLAIMER_ADDENDUM
        )

    return {
        "modality": "cardiac_ct",
        "findings": findings_out,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures_payload,
        "confidence": conf_out,
        "models_used": models_used,
        "disclaimer": disc_out,
    }


def _build_cardiac_findings(
    segments: list,
    aorta: dict,
    cac_results: dict,
    cardiac_heur: dict,
    degraded: bool,
    mr_study: bool = False,
    film_photo: bool = False,
) -> list:
    findings: list = []

    if film_photo:
        findings.append(
            Finding(
                label="Film photo input (mobile photos of printed CT/MRI film)",
                description=(
                    "Cardiac analysis from phone photographs of printed films — not DICOM. "
                    "Calcium scoring, chamber volumes, and segmentation are not reliable; obtain digital studies for definitive assessment."
                ),
                severity="warning",
                confidence=100.0,
                region="Heart",
            )
        )

    if mr_study:
        findings.append(
            Finding(
                label="MRI processed by CT-optimised pipeline",
                description=(
                    "This cardiac study was analysed with CT-optimised TotalSegmentator tasks. "
                    "If the source is MRI, results may be inaccurate — use dedicated cardiac MRI review."
                ),
                severity="warning",
                confidence=100.0,
                region="Heart",
            )
        )

    if degraded:
        findings.append(
            Finding(
                label="Degraded Input — Single Slice",
                description=(
                    "Thin or single-slice input. Cardiac chamber segmentation is most reliable "
                    "on a full CT volume."
                ),
                severity="warning",
                confidence=100.0,
                region="Heart",
            )
        )

    # CAC findings
    if cac_results.get("cac_available"):
        cac_score = cac_results.get("cac_agatston_score")
        cac_cat = cac_results.get("cac_risk_category", "unknown")
        if cac_score is not None and cac_score > 0:
            sev = "critical" if cac_cat == "severe" else ("warning" if cac_cat == "moderate" else "info")
            findings.append(
                Finding(
                    label=f"Coronary calcium (Agatston): {cac_cat}",
                    description=f"Estimated Agatston score: {cac_score:.1f} (from non-gated CT; approximate).",
                    severity=sev,
                    confidence=70.0,
                    region="Heart",
                )
            )
        else:
            findings.append(
                Finding(
                    label="Coronary calcium assessment",
                    description="No significant coronary calcium detected by HU threshold method.",
                    severity="clear",
                    confidence=70.0,
                    region="Heart",
                )
            )
    else:
        findings.append(
            Finding(
                label="Coronary calcium and wall assessment",
                description=(
                    "Coronary artery calcium score and wall-motion or wall-thickness assessment "
                    "were not performed — dedicated cardiac CT protocol and reporting are required."
                ),
                severity="info",
                confidence=100.0,
                region="Heart",
            )
        )

    # Pericardial effusion findings
    peri = cardiac_heur.get("pericardial", {})
    if peri.get("available") and peri.get("pericardial_effusion_suspected"):
        findings.append(
            Finding(
                label="Pericardial effusion suspected",
                description=f"Pericardial fluid volume: {peri.get('pericardial_effusion_volume_ml', 'unknown')} ml. Correlate clinically.",
                severity="warning",
                confidence=65.0,
                region="Heart",
            )
        )

    # Cardiomegaly findings
    cardio = cardiac_heur.get("cardiomegaly", {})
    if cardio.get("available") and cardio.get("cardiomegaly_suspected"):
        findings.append(
            Finding(
                label="Cardiomegaly suspected",
                description=f"Cardiomegaly index: {cardio.get('cardiomegaly_index', 'unknown')} (threshold >0.35). Consider clinical correlation.",
                severity="warning",
                confidence=60.0,
                region="Heart",
            )
        )

    aorta_mm = aorta.get("max_aorta_diameter_mm")
    if aorta_mm is not None and aorta_mm > 0:
        sev = "critical" if aorta.get("aaa_risk_flag") else ("warning" if aorta.get("aaa_detected") else "clear")
        findings.append(
            Finding(
                label="Aorta measurement (TotalSeg mask)",
                description=f"Estimated max aortic diameter {aorta_mm:.1f} mm (mask-based proxy).",
                severity=sev,
                confidence=85.0,
                region="Aorta",
            )
        )

    if segments:
        findings.append(
            Finding(
                label="Cardiac structures segmented",
                description=f"Structures: {', '.join(segments[:10])}{'...' if len(segments) > 10 else ''}",
                severity="info",
                confidence=85.0,
                region="Heart",
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
    
    # Parse patient_context JSON if provided
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
        return run_pipeline(fp, jid, series_dir=series_dir, source_modality="", patient_context=ctx_dict)
    finally:
        try:
            os.unlink(fp)
        except OSError:
            pass
