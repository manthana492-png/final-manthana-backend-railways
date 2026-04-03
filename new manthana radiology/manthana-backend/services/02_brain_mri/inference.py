"""
Manthana — Brain MRI Inference
TotalSegmentator total_mr + SynthSeg (subprocess) + optional Prima pipeline.
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
from disclaimer import DISCLAIMER
from prima_pipeline import run_prima_study
from schemas import Finding
from synthseg_runner import run_synthseg
from totalseg_runner import get_totalseg_version, run_totalseg, structure_list_from_result

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
    v = (os.environ.get("MRI_NARRATIVE_POLICY", "kimi_then_anthropic") or "kimi_then_anthropic").strip().lower()
    allowed = frozenset({"kimi_then_anthropic", "kimi_only", "anthropic_only", "off"})
    return v if v in allowed else "kimi_then_anthropic"


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
    for pattern in ("*.nii.gz", "*.nii", "*.dcm"):
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
    s = (notes or "").strip()
    if not s:
        return {}
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else {}
    except json.JSONDecodeError:
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


def _kimi_extra_body_mri(model: str) -> dict | None:
    m = model.lower()
    if "kimi-k2" in m:
        return {"thinking": {"type": "disabled"}}
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
        f"Patient: {age}y {sex}, clinical history: {ch}\n\n"
        f"PATIENT_CONTEXT JSON:\n{patient_json}\n\n"
        f"{india_block}\n\n"
        "Generate a concise brain MRI radiology-style narrative with India-specific differentials. "
        "Ground every statement in the JSON scores and findings; do not invent lesions or numbers."
    )


def _anthropic_mri_narrative(system: str, user_text: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return ""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        msg = client.messages.create(
            model=model,
            max_tokens=1500,
            system=system[:20000],
            messages=[{"role": "user", "content": user_text[:120000]}],
        )
        block = msg.content[0]
        return getattr(block, "text", str(block)).strip()
    except ImportError:
        logger.warning("anthropic package not installed; skip MRI Anthropic narrative")
        return ""
    except Exception as e:
        logger.warning("MRI Anthropic narrative failed: %s", e)
        return ""


def _kimi_mri_openai_narrative(
    *,
    api_key: str,
    base_url: str,
    model: str,
    image_b64: str | None,
    system: str,
    user_text: str,
    use_vision: bool,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; skip Kimi MRI narrative")
        return ""

    client = OpenAI(api_key=api_key, base_url=base_url)
    extra = _kimi_extra_body_mri(model)

    if use_vision and image_b64:
        try:
            create_kw: dict = {
                "model": model,
                "max_tokens": 1600,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
            }
            if extra is not None:
                create_kw["extra_body"] = extra
            r = client.chat.completions.create(**create_kw)
            out = (r.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception as e:
            logger.warning("Kimi vision MRI narrative failed (%s), trying text-only: %s", model, e)

    try:
        create_kw = {
            "model": model,
            "max_tokens": 1600,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        }
        if extra is not None:
            create_kw["extra_body"] = extra
        r = client.chat.completions.create(**create_kw)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Kimi text MRI narrative failed: %s", e)
        return ""


def _call_mri_narrative(
    *,
    pathology_scores: dict,
    patient_context: dict,
    image_b64: str | None,
    prompt_path: Path,
    impression: str,
    findings: list[dict],
) -> tuple[str, list[str]]:
    """
    Policy: MRI_NARRATIVE_POLICY — kimi_then_anthropic | kimi_only | anthropic_only | off.
    Vision: MRI_NARRATIVE_VISION — when disabled, Kimi is text-only.
    Never raises; returns ("", []) if all fail or policy is off.
    """
    tags: list[str] = []
    policy = _mri_narrative_policy()
    if policy == "off":
        return "", []

    system = _read_brain_mri_system_prompt(prompt_path)
    user_text = _mri_narrative_user_text(pathology_scores, impression, patient_context, findings)

    kimi_key = (
        os.environ.get("KIMI_API_KEY", "").strip()
        or os.environ.get("MOONSHOT_API_KEY", "").strip()
    )
    kimi_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1").strip()
    kimi_model = os.environ.get("KIMI_MRI_MODEL", "moonshot-v1-8k").strip()

    def _try_kimi() -> tuple[str, list[str]]:
        if not kimi_key:
            return "", []
        if image_b64 and _mri_narrative_vision_enabled():
            txt = _kimi_mri_openai_narrative(
                api_key=kimi_key,
                base_url=kimi_url,
                model=kimi_model,
                image_b64=image_b64,
                system=system,
                user_text=user_text,
                use_vision=True,
            )
            if txt:
                return txt, ["Kimi-narrative-MRI"]
        txt = _kimi_mri_openai_narrative(
            api_key=kimi_key,
            base_url=kimi_url,
            model=kimi_model,
            image_b64=None,
            system=system,
            user_text=user_text,
            use_vision=False,
        )
        if txt:
            return txt, ["Kimi-narrative-MRI"]
        return "", []

    if policy == "anthropic_only":
        ant = _anthropic_mri_narrative(system, user_text)
        if ant:
            return ant, ["Anthropic-narrative-MRI"]
        return "", []

    if policy == "kimi_only":
        return _try_kimi()

    # kimi_then_anthropic
    kt, kt_tags = _try_kimi()
    if kt:
        return kt, kt_tags
    ant = _anthropic_mri_narrative(system, user_text)
    if ant:
        return ant, ["Anthropic-narrative-MRI"]
    return "", []


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
    prima_cfg = os.getenv("PRIMA_CONFIG_YAML", "").strip()
    prima_ok = bool(prima_cfg and os.path.isfile(prima_cfg))
    ready = totalseg_ok
    full = totalseg_ok and synthseg_ok and prima_ok
    return {
        "totalseg": totalseg_ok,
        "synthseg": synthseg_ok,
        "prima": prima_ok,
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

    volume = _load_brain_volume(filepath)
    degraded_2d = _is_degraded_input(volume, filepath)

    totalseg_in = _totalseg_input_path(filepath, series_dir, job_id)
    nifti_for_synth = _resolve_nifti_for_synthseg(filepath, series_dir, job_id)
    if (not nifti_for_synth or not os.path.isfile(nifti_for_synth)) and totalseg_in.lower().endswith(
        (".nii", ".nii.gz")
    ):
        nifti_for_synth = totalseg_in

    tot_names, tot_volumes, tot_ok = _run_totalseg_mr(totalseg_in, volume)
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

    findings = _build_brain_findings(
        tot_ok=tot_ok,
        tot_names=tot_names,
        synth=synth,
        prima=prima,
        degraded_2d=degraded_2d,
        filepath=filepath,
        clinical_notes=notes,
    )
    impression = _build_impression(pathology_scores, tot_ok, synth, prima)
    models_used = _models_used(tot_ok, synth, prima)

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
    need_vision_slice = _mri_narrative_vision_enabled() and _pol not in ("off", "anthropic_only")
    slice_b64 = _volume_middle_axial_b64_png(volume) if need_vision_slice else None
    llm_narr, narr_tags = _call_mri_narrative(
        pathology_scores=pathology_scores,
        patient_context=patient_ctx,
        image_b64=slice_b64,
        prompt_path=_BRAIN_MRI_SYSTEM,
        impression=impression,
        findings=findings_payload,
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

    return {
        "job_id": job_id,
        "modality": "brain_mri",
        "findings": findings_out,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": confidence,
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }


def _is_degraded_input(volume: np.ndarray, filepath: str) -> bool:
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


def _run_totalseg_mr(filepath: str, volume: np.ndarray) -> tuple[list, dict, bool]:
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
    degraded_2d: bool,
    filepath: str,
    clinical_notes: str,
) -> list[Finding]:
    out: list[Finding] = []
    pl = filepath.lower()

    if pl.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
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
    if not parts:
        return "Brain MRI analysis complete. Clinical correlation recommended."
    return " ".join(parts) + " Clinical correlation recommended."


def _models_used(tot_ok: bool, synth: dict, prima: dict) -> list[str]:
    mu = []
    if tot_ok:
        mu.append("TotalSegmentator-MRI")
    if synth.get("available"):
        mu.append("SynthSeg")
    if prima.get("available"):
        mu.append("Prima")
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
        sd = os.path.abspath(filepath)
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
    ext = os.path.splitext(filename_hint)[1] if filename_hint else ".png"
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
