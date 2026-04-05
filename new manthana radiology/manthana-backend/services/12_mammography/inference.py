"""
Manthana Mammography Engine — Lab-Rasool/Mirai (MIT), 4-view risk scoring.
Single-image uploads: BI-RADS-oriented heuristics + Kimi → Claude narrative.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _find_shared() -> Path:
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
    raise RuntimeError("Cannot find shared/ directory for mammography service.")


_shared = _find_shared()
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

try:
    from env_file import load_api_keys_env

    load_api_keys_env()
except ImportError:
    pass

logger = logging.getLogger("manthana.mammography")

PROMPT_PATH = Path(__file__).parent / "prompts" / "mammography_system.txt"
DEVICE = os.environ.get("DEVICE", "cpu")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
MIRAI_CACHE = MODEL_DIR / "mirai_cache"
MIRAI_REPO = os.environ.get("MIRAI_HF_REPO", "Lab-Rasool/Mirai")

DISCLAIMER = (
    "AI-assisted mammography second opinion. Requires radiologist confirmation. "
    "Mirai risk model validated on US/Swedish/Taiwanese populations — "
    "independent Indian-population validation has not been published. "
    "Not a certified medical device. Not a replacement for radiologist review."
)

VIEW_KEYS = frozenset({"L-CC", "L-MLO", "R-CC", "R-MLO"})

# Mirai model card: 5-year risk thresholds (percent)
_mirai_model: Any = None
_mirai_preprocessor: Any = None
_mirai_model_dir: Optional[str] = None


def is_loaded() -> bool:
    return _mirai_model is not None


def _risk_category(five_yr_prob: float) -> str:
    pct = five_yr_prob * 100.0
    if pct < 1.67:
        return "low"
    if pct < 3.0:
        return "average"
    if pct < 5.0:
        return "moderate"
    return "high"


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _load_mirai() -> bool:
    global _mirai_model, _mirai_preprocessor, _mirai_model_dir
    if _mirai_model is not None:
        return True
    try:
        import torch
        from huggingface_hub import snapshot_download

        tok = _hf_token()
        model_dir = snapshot_download(
            repo_id=MIRAI_REPO,
            cache_dir=str(MIRAI_CACHE),
            token=tok,
        )
        _mirai_model_dir = model_dir
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from configuration_mirai import MiraiConfig  # type: ignore
        from modeling_mirai import MiraiModel  # type: ignore
        from preprocessor import MiraiPreprocessor  # type: ignore

        config = MiraiConfig.from_pretrained(model_dir)
        _mirai_model = MiraiModel.from_pretrained(model_dir, config=config)
        _mirai_model.eval()
        if DEVICE == "cuda":
            _mirai_model = _mirai_model.cuda()
        _mirai_preprocessor = MiraiPreprocessor()
        return True
    except Exception as e:
        logger.warning("Mirai load failed: %s", e)
        return False


def _dicom_to_png_for_mirai(dicom_path: str, output_path: str) -> str:
    import numpy as np
    import pydicom
    from PIL import Image

    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype("float64")

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if isinstance(wc, (list, tuple)):
            wc = wc[0]
        if isinstance(ww, (list, tuple)):
            ww = ww[0]
        lo = float(wc) - float(ww) / 2.0
        hi = float(wc) + float(ww) / 2.0
        arr = np.clip(arr, lo, hi)

    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 65535.0
    arr_u16 = arr.astype(np.uint16)

    img = Image.fromarray(arr_u16, mode="I;16")
    # Mirai card: 1664×2048 (H×W) — PIL size is (W, H)
    if img.size != (2048, 1664):
        img = img.resize((2048, 1664), Image.LANCZOS)

    img_8 = Image.fromarray((np.asarray(img, dtype=np.uint32) // 256).astype(np.uint8))
    img_8.save(output_path)
    return output_path


def _prepare_view_pngs(views: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    import pydicom

    png_paths: Dict[str, str] = {}
    tmp_files: List[str] = []
    for view_key, path in views.items():
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"View file missing: {view_key}={path}")
        is_dicom = False
        try:
            pydicom.dcmread(path, stop_before_pixels=True)
            is_dicom = True
        except Exception:
            is_dicom = False
        if is_dicom:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            _dicom_to_png_for_mirai(path, tmp.name)
            png_paths[view_key] = tmp.name
            tmp_files.append(tmp.name)
        else:
            png_paths[view_key] = path
    return png_paths, tmp_files


def _run_mirai(views: Dict[str, str], patient_context: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    if not _load_mirai():
        return {"available": False, "reason": "Mirai model failed to load"}

    png_views: Dict[str, str]
    tmp_files: List[str]
    try:
        png_views, tmp_files = _prepare_view_pngs(views)
    except Exception as e:
        return {"available": False, "reason": str(e)}

    try:
        risk_factors = {
            "age": int(patient_context.get("age", 50)),
            "density": int(patient_context.get("birads_density", 2)),
            "family_history": bool(patient_context.get("family_history", False)),
            "biopsy_benign": bool(patient_context.get("biopsy_benign", False)),
            "biopsy_lcis": bool(patient_context.get("biopsy_lcis", False)),
            "biopsy_atypical": bool(patient_context.get("biopsy_atypical", False)),
            "menarche_age": int(patient_context.get("menarche_age", 13)),
            "menopause_age": int(patient_context.get("menopause_age", 0)),
            "first_pregnancy_age": int(patient_context.get("first_pregnancy_age", 0)),
            "race": int(patient_context.get("race", 1)),
            "weight": float(patient_context.get("weight_kg", 60)),
            "height": float(patient_context.get("height_cm", 160)),
            "parous": bool(patient_context.get("parous", False)),
            "menopausal_status": int(patient_context.get("menopausal_status", 0)),
        }

        assert _mirai_preprocessor is not None and _mirai_model is not None
        exam_tensor = _mirai_preprocessor.load_mammogram_exam(png_views)
        risk_factors_tensor = _mirai_preprocessor.prepare_risk_factors(risk_factors)

        exam_tensor = exam_tensor.permute(1, 0, 2, 3)
        if DEVICE == "cuda":
            exam_tensor = exam_tensor.cuda()
            risk_factors_tensor = risk_factors_tensor.cuda()

        batch_images = exam_tensor.unsqueeze(0)
        batch_risk_factors = risk_factors_tensor.unsqueeze(0)
        dev = next(_mirai_model.parameters()).device
        batch_metadata = {
            "time_seq": torch.zeros(1, 4, dtype=torch.long, device=dev),
            "view_seq": torch.tensor([[0, 1, 0, 1]], dtype=torch.long, device=dev),
            "side_seq": torch.tensor([[0, 0, 1, 1]], dtype=torch.long, device=dev),
        }

        with torch.no_grad():
            outputs = _mirai_model(
                images=batch_images,
                risk_factors=batch_risk_factors,
                batch_metadata=batch_metadata,
                return_dict=True,
            )

        if hasattr(outputs, "probabilities") and outputs.probabilities is not None:
            probs = outputs.probabilities[0].detach().cpu().numpy()
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            probs = torch.sigmoid(outputs.logits[0]).detach().cpu().numpy()
        else:
            return {"available": False, "reason": "Unknown Mirai output format"}

        probs = probs.flatten()
        if len(probs) < 5:
            return {"available": False, "reason": "Unexpected Mirai output shape"}

        five_yr = float(probs[4])
        return {
            "available": True,
            "cancer_risk_1yr": round(float(probs[0]), 5),
            "cancer_risk_2yr": round(float(probs[1]), 5),
            "cancer_risk_3yr": round(float(probs[2]), 5),
            "cancer_risk_5yr": round(five_yr, 5),
            "risk_category": _risk_category(five_yr),
            "is_high_risk": 1.0 if five_yr > 0.05 else 0.0,
            "views_used": list(views.keys()),
        }
    except Exception as e:
        logger.exception("Mirai inference failed: %s", e)
        return {"available": False, "reason": str(e)}
    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _build_findings(mirai_scores: Dict[str, Any], has_four_views: bool) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    if not has_four_views:
        findings.append(
            {
                "label": "Incomplete exam — risk model not run",
                "severity": "warning",
                "confidence": 0.0,
                "description": (
                    "Mirai requires all 4 standard views: L-CC, L-MLO, R-CC, R-MLO. "
                    "Single-image upload produces visual assessment only — "
                    "no validated quantitative risk scores are available."
                ),
            }
        )
        findings.append(
            {
                "label": "AI visual assessment only",
                "severity": "info",
                "confidence": 0.0,
                "description": "Narrative report based on image appearance when available. Radiologist review required.",
            }
        )
        return findings

    if not mirai_scores.get("available"):
        findings.append(
            {
                "label": "Risk model unavailable",
                "severity": "warning",
                "confidence": 0.0,
                "description": mirai_scores.get("reason", "Mirai model loading or inference failed"),
            }
        )
        return findings

    risk_5yr = float(mirai_scores["cancer_risk_5yr"])
    category = str(mirai_scores["risk_category"])
    severity_map = {
        "high": "critical",
        "moderate": "warning",
        "average": "info",
        "low": "clear",
    }
    severity = severity_map.get(category, "info")

    findings.append(
        {
            "label": f"5-year breast cancer risk: {category.upper()} ({risk_5yr * 100:.2f}%)",
            "severity": severity,
            "confidence": float(min(99.0, risk_5yr * 1000.0)),
            "description": (
                f"Mirai model (MIT, Yala et al.). "
                f"1yr: {mirai_scores['cancer_risk_1yr'] * 100:.2f}% | "
                f"2yr: {mirai_scores['cancer_risk_2yr'] * 100:.2f}% | "
                f"5yr: {risk_5yr * 100:.2f}%"
            ),
        }
    )

    threshold_note = {
        "high": "5yr >5% — discuss risk-reduction strategies; consider supplemental MRI for dense breasts",
        "moderate": "5yr 3–5% — consider supplemental screening; short-interval follow-up per protocol",
        "average": "5yr 1.67–3% — annual mammography per local guidelines",
        "low": "5yr <1.67% — standard screening interval",
    }
    findings.append(
        {
            "label": "Risk-aligned management suggestion",
            "severity": "info",
            "confidence": 0.0,
            "description": threshold_note.get(category, "Correlate clinically."),
        }
    )

    findings.append(
        {
            "label": "Population validation note",
            "severity": "info",
            "confidence": 0.0,
            "description": (
                "Mirai was validated on US, Swedish, and Taiwanese cohorts. "
                "No published independent validation for Indian women. "
                "Indian breast cancer often presents at younger ages; interpret scores with clinical judgment."
            ),
        }
    )
    return findings


def _gaussian_blur2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy import ndimage

        return ndimage.gaussian_filter(arr, sigma=sigma)
    except Exception:
        return arr


def _heuristic_mammo_from_gray(gray: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Single-view radiomics-lite heuristics for BI-RADS-shaped output (not a certified CAD)."""
    g = np.asarray(gray, dtype=np.float32)
    if g.size == 0:
        return _default_mammo_structures_scores()
    g = (g - g.min()) / (max(float(g.max() - g.min()), 1e-6))
    h, w = g.shape[:2]
    mean_all = float(np.mean(g))
    if mean_all > 0.55:
        density = "ACR_C"
        density_score = 0.72
    elif mean_all > 0.38:
        density = "ACR_B"
        density_score = 0.45
    elif mean_all > 0.22:
        density = "ACR_C"
        density_score = 0.55
    else:
        density = "ACR_A"
        density_score = 0.25

    u0, u1 = int(h * 0.15), int(h * 0.55)
    v0, v1 = int(w * 0.35), int(w * 0.65)
    roi = g[u0:u1, v0:v1]
    roi_mean = float(np.mean(roi)) if roi.size else mean_all

    bright = (g > 0.92).astype(np.uint8)
    try:
        from scipy import ndimage

        labeled, nlab = ndimage.label(bright)
        sizes = [np.sum(labeled == k) for k in range(1, nlab + 1)]
        n_small = sum(1 for s in sizes if 4 <= s <= 400)
    except Exception:
        n_small = int(np.sum(bright) / 50)

    calc_present = n_small >= 8
    calc_morph = "fine_pleomorphic" if calc_present else None
    calc_dist = "grouped" if calc_present else None

    edge = np.abs(g - _gaussian_blur2d(g, 2.0))
    edge_uoq = float(np.mean(edge[int(h * 0.1) : int(h * 0.45), int(w * 0.25) : int(w * 0.55)]))

    # Avoid flagging noisy uniform mammo simulations as masses (high FP on random texture).
    mass_present = edge_uoq > 0.075 or roi_mean > 0.68
    mass_shape = "irregular" if mass_present else "oval"
    mass_margin = "spiculated" if edge_uoq > 0.055 else ("indistinct" if mass_present else "circumscribed")
    mass_loc = "upper_outer_quadrant" if mass_present else None

    birads = "1"
    arch = edge_uoq > 0.05 and not mass_present
    asymm = False
    axillary = False

    if not mass_present and not calc_present and edge_uoq < 0.06:
        birads = "1"
    elif calc_present and not mass_present:
        birads = "3"
    elif mass_present and mass_margin == "spiculated":
        birads = "4B"
    elif mass_present:
        birads = "4A"

    mal = 0.12
    mass_conf = 0.15
    calc_conf = 0.08
    b4plus = 0.1

    if birads == "1":
        mal = 0.1
        b4plus = 0.08
        is_crit = False
    elif birads in ("4A", "4B", "4C", "5"):
        mal = 0.62 if birads == "4B" else 0.48
        mass_conf = 0.72 if mass_present else 0.4
        b4plus = 0.85
        is_crit = birads in ("4C", "5") or (
            birads == "4B" and mass_margin == "spiculated" and mass_present
        )
    if calc_present:
        calc_conf = min(0.9, 0.25 + n_small * 0.02)
    if birads == "3":
        b4plus = 0.22
        mal = 0.18
        is_crit = False

    structures: Dict[str, Any] = {
        "view": "MLO",
        "breast_density": density,
        "birads_category": birads,
        "mass_present": bool(mass_present),
        "mass_location": mass_loc,
        "mass_shape": mass_shape,
        "mass_margin": mass_margin,
        "calcification_present": bool(calc_present),
        "calcification_morphology": calc_morph,
        "calcification_distribution": calc_dist,
        "asymmetry_present": asymm,
        "architectural_distortion": bool(arch),
        "axillary_adenopathy": axillary,
        "is_critical": bool(is_crit),
        "narrative_report": "",
    }
    scores: Dict[str, float] = {
        "malignancy_confidence": round(mal, 4),
        "mass_confidence": round(mass_conf, 4),
        "calcification_confidence": round(calc_conf, 4),
        "birads_4_or_above": round(b4plus, 4),
        "density_score": round(density_score, 4),
    }
    return structures, scores


def _default_mammo_structures_scores() -> Tuple[Dict[str, Any], Dict[str, float]]:
    st, sc = _heuristic_mammo_from_gray(np.ones((64, 64), dtype=np.float32) * 0.4)
    return st, sc


def _heuristic_mammo_from_path(filepath: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
    from PIL import Image

    try:
        img = Image.open(filepath).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        return _heuristic_mammo_from_gray(arr)
    except Exception as e:
        logger.warning("Mammo heuristic load failed: %s", e)
        return _default_mammo_structures_scores()


def _mammo_narrative_openrouter(
    structures: Dict[str, Any],
    pathology_scores: Dict[str, float],
    patient_context: Dict[str, Any],
    mirai_scores: Dict[str, Any],
    image_b64: str,
    has_four_views: bool,
) -> Tuple[str, List[str]]:
    tags: List[str] = []
    scores = {k: float(pathology_scores.get(k, 0) or 0) for k in (
        "malignancy_confidence",
        "mass_confidence",
        "calcification_confidence",
        "birads_4_or_above",
        "density_score",
    )}
    age = patient_context.get("age", "")
    sex = patient_context.get("sex", "")
    clin = patient_context.get("clinical_history", patient_context.get("history", ""))
    fam = patient_context.get("family_history", "unknown")
    hosp = patient_context.get("hospital", "India")
    mirai_block = json.dumps(
        {"mirai_risk_scores": mirai_scores, "has_four_views": has_four_views},
        indent=2,
        default=str,
    )[:8000]
    user_text = f"""Mammography Analysis Results:
View: {structures.get('view')}
Breast density: {structures.get('breast_density')}
BI-RADS category: {structures.get('birads_category')}
Mass: present={structures.get('mass_present')}, shape={structures.get('mass_shape')}, margin={structures.get('mass_margin')}
Calcifications: present={structures.get('calcification_present')}, morphology={structures.get('calcification_morphology')}
Architectural distortion: {structures.get('architectural_distortion')}
Axillary adenopathy: {structures.get('axillary_adenopathy')}

Pathology scores:
  Malignancy confidence: {scores.get('malignancy_confidence', 0):.3f}
  Mass confidence: {scores.get('mass_confidence', 0):.3f}
  BI-RADS 4+ confidence: {scores.get('birads_4_or_above', 0):.3f}
  Density score: {scores.get('density_score', 0):.3f}

Patient: {age}y {sex}
Clinical: {clin}
Family history: {fam}
Hospital: {hosp}

MIRAI / CONTEXT:
{mirai_block}

Generate a structured BI-RADS mammography report.
Include: ACR density classification, full lesion descriptors,
BI-RADS category with reasoning, India-specific context
(Kidwai/Tata Memorial/NRGCP if relevant), recommended action."""
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        system_prompt = (
            "You are a senior breast radiologist writing BI-RADS structured reports for Indian hospitals."
        )
    try:
        from llm_router import llm_router

        mime = _sniff_media_type(image_b64) if image_b64 else "image/jpeg"
        if image_b64:
            out = llm_router.complete_for_role(
                "mammography",
                system_prompt,
                user_text,
                image_b64=image_b64,
                image_mime=mime,
                max_tokens=2000,
            )
            txt = (out.get("content") or "").strip()
            if txt:
                tags.append("OpenRouter-narrative-Mammo")
                return txt, tags
        out = llm_router.complete_for_role(
            "mammography",
            system_prompt,
            user_text,
            max_tokens=2000,
        )
        txt = (out.get("content") or "").strip()
        if txt:
            tags.append("OpenRouter-narrative-Mammo")
            return txt, tags
    except Exception as e:
        logger.warning("OpenRouter mammo narrative failed: %s", e)
    return "", tags


def _safe_scores(raw: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            if isinstance(v, bool):
                out[k] = 1.0 if v else 0.0
            else:
                out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def _sniff_media_type(b64: str) -> str:
    try:
        h = base64.b64decode(b64[:32] + "==")
    except (binascii.Error, ValueError):
        return "image/jpeg"
    if h[:2] == b"\xff\xd8":
        return "image/jpeg"
    if h[:4] == b"\x89PNG":
        return "image/png"
    return "image/jpeg"


def _apply_mirai_to_birads(
    structures: Dict[str, Any],
    scores: Dict[str, float],
    mirai_scores: Dict[str, Any],
) -> None:
    if not mirai_scores.get("available"):
        return
    cat = str(mirai_scores.get("risk_category", "")).lower()
    five = float(mirai_scores.get("cancer_risk_5yr", 0) or 0)
    if five >= 0.05 or cat == "high":
        structures["birads_category"] = "4B"
        scores["birads_4_or_above"] = max(float(scores.get("birads_4_or_above", 0) or 0), 0.88)
        scores["malignancy_confidence"] = max(float(scores.get("malignancy_confidence", 0) or 0), 0.75)
        structures["is_critical"] = True
    elif five >= 0.03 or cat == "moderate":
        if str(structures.get("birads_category", "1")) in ("1", "2"):
            structures["birads_category"] = "3"
        scores["birads_4_or_above"] = max(float(scores.get("birads_4_or_above", 0) or 0), 0.35)


def run_pipeline(
    filepath: str,
    job_id: str = "",
    patient_context: Optional[Dict[str, Any]] = None,
    claude_client: Any = None,
    image_b64: str = "",
) -> Dict[str, Any]:
    if patient_context is None:
        patient_context = {}

    if not image_b64:
        with open(filepath, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")

    raw_views = patient_context.get("views")
    views: Dict[str, str] = raw_views if isinstance(raw_views, dict) else {}
    has_four_views = VIEW_KEYS.issubset(set(views.keys()))

    if has_four_views:
        mirai_scores = _run_mirai(views, patient_context)
    else:
        mirai_scores = {
            "available": False,
            "reason": "Four standard views required — single upload cannot produce Mirai risk scores",
            "views_provided": list(views.keys()) if views else ["single_image"],
        }

    h_struct, h_scores = _heuristic_mammo_from_path(filepath)
    _apply_mirai_to_birads(h_struct, h_scores, mirai_scores)

    findings = _build_findings(mirai_scores, has_four_views)

    scores_raw: Dict[str, Any] = dict(h_scores)
    if mirai_scores.get("available"):
        scores_raw.update(
            {
                "cancer_risk_1yr": mirai_scores["cancer_risk_1yr"],
                "cancer_risk_2yr": mirai_scores["cancer_risk_2yr"],
                "cancer_risk_3yr": mirai_scores["cancer_risk_3yr"],
                "cancer_risk_5yr": mirai_scores["cancer_risk_5yr"],
                "is_high_risk": mirai_scores["is_high_risk"],
            }
        )

    narrative, narr_tags = _mammo_narrative_openrouter(
        structures=h_struct,
        pathology_scores=scores_raw,
        patient_context=patient_context,
        mirai_scores=mirai_scores,
        image_b64=image_b64,
        has_four_views=has_four_views,
    )
    h_struct["narrative_report"] = narrative

    if narrative and not narrative.startswith("[Narrative unavailable"):
        impression = narrative[:300].strip()
    elif mirai_scores.get("available"):
        impression = (
            f"5-year breast cancer risk category: {mirai_scores.get('risk_category', 'unknown')}. "
            "Radiologist review required."
        )
    else:
        impression = (
            "Single-image upload — Mirai risk model not run. "
            "Heuristic BI-RADS-style metrics below; radiologist review required."
        )

    birads_s = str(h_struct.get("birads_category", "1"))
    is_critical = bool(h_struct.get("is_critical"))
    if birads_s in ("4B", "4C", "5", "6"):
        is_critical = True
    if birads_s == "4A" and float(scores_raw.get("birads_4_or_above", 0) or 0) > 0.75:
        is_critical = True
    if mirai_scores.get("available") and float(mirai_scores.get("is_high_risk", 0) or 0) > 0.5:
        is_critical = True

    structures: Dict[str, Any] = {
        **h_struct,
        "has_four_views": has_four_views,
        "risk_category": mirai_scores.get("risk_category", "unknown"),
        "views_used": mirai_scores.get("views_used", []),
        "patient_context": patient_context,
        "india_note": (
            "Mirai was not validated on Indian populations. "
            "Breast cancer often presents younger in India than in Western cohorts. "
            "Risk scores are supplemental only."
        ),
    }
    structures["is_critical"] = is_critical

    models_used: List[str] = []
    if mirai_scores.get("available"):
        models_used.append("Mirai")
    models_used.append("heuristic_birads_cv")
    if "OpenRouter-narrative-Mammo" in narr_tags:
        models_used.append("OpenRouter-mammography")

    return {
        "modality": "mammography",
        "findings": findings,
        "impression": impression,
        "pathology_scores": _safe_scores(scores_raw),
        "structures": structures,
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
        "confidence": "medium",
        "job_id": job_id,
        "is_critical": is_critical,
    }


def run_mammography_pipeline_b64(
    image_b64: str,
    patient_context: Optional[Dict[str, Any]] = None,
    claude_client: Any = None,
    patient_context_json: Optional[str] = None,
) -> Dict[str, Any]:
    if patient_context is None:
        patient_context = {}
    if patient_context_json and str(patient_context_json).strip():
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                patient_context = {**patient_context, **parsed}
        except json.JSONDecodeError:
            pass
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except (binascii.Error, ValueError) as e:
        return {
            "available": False,
            "reason": "invalid_base64",
            "message": str(e),
        }
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        return run_pipeline(
            filepath=tmp,
            job_id="",
            patient_context=patient_context,
            image_b64=image_b64,
        )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
