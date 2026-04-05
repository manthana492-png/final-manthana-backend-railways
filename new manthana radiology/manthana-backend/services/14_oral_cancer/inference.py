"""
Manthana — Oral cancer inference (India-focused, production-safe).

Paths (graceful degradation, no training in-repo):
  A) Clinical photo: EfficientNet-B3 + fine-tuned checkpoint (legacy) if present.
  B) Clinical photo: optional torchvision EfficientNet-V2-M weights file if present.
  C) Histopathology / H&E-style: UNI encoder (HF) + optional linear head; heuristic if no head.
  D) Vision LLM (OpenRouter; SSOT config/cloud_inference.yaml) structured JSON fallback when local paths fail or are unavailable.

Never raises except OralServiceUnavailableError when ORAL_CANCER_ENABLED is false.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image as PILImage

_here_inf = os.path.dirname(os.path.abspath(__file__))
for _p in (
    "/app/shared",
    os.path.normpath(os.path.join(_here_inf, "..", "..", "shared")),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

from transformers import EfficientNetForImageClassification, EfficientNetImageProcessor

from model_loader import LazyModel
from disclaimer import DISCLAIMER
from config import (
    CHECKPOINT_FILENAME,
    DEVICE,
    EFFNET_V2M_CHECKPOINT,
    MODEL_DIR,
    ORAL_CANCER_ENABLED,
    UNI_HEAD_CHECKPOINT,
    UNI_MODEL_ID,
)
from schemas import Finding

logger = logging.getLogger("manthana.oral_cancer")

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PATH = PROMPT_DIR / "oral_vision_fallback.txt"
ORAL_CANCER_SYSTEM_PATH = PROMPT_DIR / "oral_cancer_system.md"

ORAL_DISCLAIMER = (
    "AI second opinion only. Tissue diagnosis / specialist review and biopsy when indicated — required."
)

CLASSES = {
    0: {"name": "Normal", "risk": "low", "description": "Healthy oral tissue"},
    1: {"name": "OPMD", "risk": "medium", "description": "Oral Potentially Malignant Disorder"},
    2: {"name": "OSCC", "risk": "high", "description": "Suspicious for Oral Squamous Cell Carcinoma"},
}

efficientnet_model = LazyModel(
    model_id="google/efficientnet-b3",
    cache_name="oral_cancer_effnet",
    device=DEVICE,
    model_class=EfficientNetForImageClassification,
    extra_kwargs={
        "num_labels": 3,
        "ignore_mismatched_sizes": True,
        "id2label": {0: "Normal", 1: "OPMD", 2: "OSCC"},
        "label2id": {"Normal": 0, "OPMD": 1, "OSCC": 2},
    },
)

_processor: EfficientNetImageProcessor | None = None
_checkpoint_applied = False


class OralServiceUnavailableError(Exception):
    """Service disabled via ORAL_CANCER_ENABLED."""


class OralClassificationFailed(Exception):
    """Deprecated for HTTP mapping — kept for tests/callers that catch explicitly."""


def _checkpoint_path() -> str:
    return os.path.join(MODEL_DIR, CHECKPOINT_FILENAME)


def _v2m_path() -> str:
    return os.path.join(MODEL_DIR, EFFNET_V2M_CHECKPOINT)


def _uni_head_path() -> str:
    return os.path.join(MODEL_DIR, UNI_HEAD_CHECKPOINT)


def _get_processor() -> EfficientNetImageProcessor:
    global _processor
    if _processor is None:
        _processor = EfficientNetImageProcessor.from_pretrained(
            "google/efficientnet-b3",
            size={"height": 300, "width": 300},
            do_center_crop=False,
            crop_size={"height": 300, "width": 300},
            cache_dir=os.path.join(MODEL_DIR, "oral_cancer_effnet"),
        )
    return _processor


def is_service_ready() -> bool:
    """Ready when service is enabled (weights optional — vision / degrade paths allowed)."""
    return ORAL_CANCER_ENABLED


def _has_b3_checkpoint() -> bool:
    return os.path.isfile(_checkpoint_path())


def _has_v2m_weights() -> bool:
    return os.path.isfile(_v2m_path())


def _has_uni_head() -> bool:
    return os.path.isfile(_uni_head_path())


def _has_openrouter_cloud() -> bool:
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"):
        k = (os.environ.get(name) or "").strip()
        if k and len(k) >= 8:
            return True
    return False


def get_loaded_status() -> dict[str, Any]:
    return {
        "ready": is_service_ready(),
        "b3_checkpoint": _has_b3_checkpoint(),
        "effnet_v2m_weights": _has_v2m_weights(),
        "uni_head": _has_uni_head(),
        "openrouter_configured": _has_openrouter_cloud(),
        "oral_cancer_enabled": ORAL_CANCER_ENABLED,
        "efficientnet_loaded": efficientnet_model.is_loaded(),
    }


def is_loaded() -> bool:
    return efficientnet_model.is_loaded()


def _apply_checkpoint_once(model: Any) -> None:
    global _checkpoint_applied
    if _checkpoint_applied:
        return
    ft_path = _checkpoint_path()
    if not os.path.isfile(ft_path):
        return
    import torch

    try:
        try:
            raw = torch.load(ft_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(ft_path, map_location="cpu")
        meta: dict[str, Any] = {}
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
            meta = {k: v for k, v in raw.items() if k != "state_dict"}
        else:
            state = raw
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "B3 checkpoint loaded: %s | missing=%d unexpected=%d meta=%s",
            ft_path,
            len(missing),
            len(unexpected),
            meta,
        )
        _checkpoint_applied = True
    except Exception as e:
        logger.exception("B3 checkpoint not applied: %s", e)


def _run_b3_classification(pil_img: PILImage.Image) -> tuple[np.ndarray, int] | None:
    if not _has_b3_checkpoint():
        return None
    import torch
    import torch.nn.functional as F

    try:
        model = efficientnet_model.get()
        model.eval()
        _apply_checkpoint_once(model)
        processor = _get_processor()
        inputs = processor(images=pil_img, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
            if not hasattr(output, "logits") or output.logits is None:
                return None
            logits = output.logits
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        probs = np.asarray(probs, dtype=np.float64).flatten()
        if probs.size < 3:
            return None
        predicted = int(np.argmax(probs[:3]))
        return probs[:3], predicted
    except Exception as e:
        logger.warning("B3 clinical path failed: %s", e)
        return None


def _run_v2m_classification(pil_img: PILImage.Image) -> tuple[np.ndarray, int] | None:
    if not _has_v2m_weights():
        return None
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.models import efficientnet_v2_m

    try:
        wpath = _v2m_path()
        model = efficientnet_v2_m(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features, 3),
        )
        try:
            state = torch.load(wpath, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(wpath, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
        model.to(device)

        tform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        x = tform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        probs = np.asarray(probs, dtype=np.float64).flatten()
        if probs.size < 3:
            return None
        pred = int(np.argmax(probs[:3]))
        return probs[:3], pred
    except Exception as e:
        logger.warning("EfficientNet-V2-M path failed: %s", e)
        return None


def _uni_embedding_and_scores(pil_img: PILImage.Image) -> dict[str, Any] | None:
    """UNI encoder + optional linear head; heuristic scores if head missing."""
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(UNI_MODEL_ID, trust_remote_code=True)
        model = AutoModel.from_pretrained(UNI_MODEL_ID, trust_remote_code=True)
        model.eval()
        device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
        model.to(device)
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            pool = out.last_hidden_state.mean(dim=1) if hasattr(out, "last_hidden_state") else out.pooler_output
            emb = pool.squeeze(0).float().cpu()
        emb_vec = emb.numpy()
        dim = emb_vec.size

        head_path = _uni_head_path()
        if os.path.isfile(head_path):
            try:
                try:
                    blob = torch.load(head_path, map_location="cpu", weights_only=True)
                except TypeError:
                    blob = torch.load(head_path, map_location="cpu")
                if isinstance(blob, dict) and "head" in blob:
                    state = blob["head"]
                    n_cls = int(blob.get("num_classes", 3))
                else:
                    state = blob
                    n_cls = 3
                lin = nn.Linear(dim, n_cls)
                lin.load_state_dict(state, strict=False)
                lin.eval()
                with torch.no_grad():
                    logits = lin(torch.from_numpy(emb_vec).unsqueeze(0))
                    probs = torch.softmax(logits, dim=1).squeeze().numpy()
                probs = np.asarray(probs, dtype=np.float64).flatten()
                if probs.size >= 3:
                    pred = int(np.argmax(probs[:3]))
                    return {
                        "probs3": probs[:3],
                        "predicted": pred,
                        "model_path": head_path,
                        "limitation": "",
                        "embedding_dim": dim,
                    }
            except Exception as e:
                logger.warning("UNI head load failed, using heuristic: %s", e)

        # Heuristic: no pathology head — weak signal from embedding energy (non-diagnostic)
        energy = float(np.linalg.norm(emb_vec))
        # Map to soft uniform-ish distribution with slight spread (explicitly non-clinical)
        base = 0.28
        spread = min(0.15, energy / (dim ** 0.5 + 1e-6) * 0.05)
        probs = np.array([base + spread, base, base - spread * 0.5], dtype=np.float64)
        probs = np.clip(probs, 0.05, 0.9)
        probs = probs / probs.sum()
        return {
            "probs3": probs,
            "predicted": int(np.argmax(probs)),
            "model_path": "",
            "limitation": (
                "UNI embedding only — no validated linear head on this deployment. "
                "Scores are non-diagnostic heuristics; histopathology is required for diagnosis."
            ),
            "embedding_dim": dim,
        }
    except Exception as e:
        logger.warning("UNI path failed: %s", e)
        return None


def _parse_json_from_text(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return json.loads(t)


def _sniff_media_type(image_bytes: bytes) -> str:
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


def _openrouter_oral_vision_json(
    image_b64: str,
    patient_context: dict[str, Any],
    clinical_notes: str,
) -> tuple[dict[str, Any] | None, str]:
    """OpenRouter vision + JSON (role oral_cancer). Returns (parsed, model_used_slug). Never raises."""
    try:
        from llm_router import llm_router
    except Exception:
        return None, ""
    system = (
        "You output only JSON as specified. Indian oral oncology context. "
        "Never claim histologic certainty from photos."
    )
    user_text = (
        PROMPT_PATH.read_text()
        + "\n\nPatient context JSON:\n"
        + json.dumps(patient_context, ensure_ascii=False)
        + "\nClinical notes:\n"
        + (clinical_notes or "")
    )
    try:
        media_type = _sniff_media_type(base64.b64decode(image_b64))
        mime = "image/png" if media_type == "image/png" else "image/jpeg"
        out = llm_router.complete_for_role(
            "oral_cancer",
            system,
            user_text,
            image_b64=image_b64,
            image_mime=mime,
            max_tokens=1200,
            requires_json=True,
        )
        txt = (out.get("content") or "").strip()
        mu = str(out.get("model_used") or "").strip()
        return _parse_json_from_text(txt), mu
    except Exception as e:
        logger.warning("OpenRouter oral vision JSON failed (soft): %s", e)
        return None, ""


def _vision_llm_oral(
    pil_img: PILImage.Image,
    image_b64: str,
    patient_context: dict[str, Any],
    clinical_notes: str,
) -> tuple[dict[str, Any] | None, str, str]:
    """Returns (parsed_json, provider, vision_model_slug). Provider is openrouter | empty."""
    _ = pil_img
    v, mu = _openrouter_oral_vision_json(image_b64, patient_context, clinical_notes)
    if v:
        return v, "openrouter", mu
    return None, "", ""


def _merge_patient_context(
    clinical_notes: str,
    patient_context: dict[str, Any] | None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = dict(patient_context or {})
    if clinical_notes and not ctx.get("clinical_notes"):
        ctx["clinical_notes"] = clinical_notes
    return ctx


def _infer_input_type(
    filepath: str,
    ctx: dict[str, Any],
    override: str | None,
) -> str:
    if override in ("clinical_photo", "histopathology", "mixed", "unknown"):
        return override
    hint = str(ctx.get("input_type", "") or "").lower()
    if hint in ("clinical_photo", "histopathology", "mixed", "unknown"):
        return hint
    low = (filepath or "").lower()
    for token in (".svs", "hist", "he_", "h&e", "biopsy", "wsi", "pathology"):
        if token in low:
            return "histopathology"
    return "clinical_photo"


def _habit_risk_from_context(ctx: dict[str, Any], notes: str) -> str:
    blob = f"{notes} {json.dumps(ctx)}".lower()
    betel = any(
        x in blob
        for x in ("betel", "gutka", "khaini", "paan", "areca", "quid", "osmf", "fibrosis")
    )
    tobacco = any(x in blob for x in ("tobacco", "smok", "cigar", "bidi", "cigarette"))
    if betel and tobacco:
        return "combined"
    if betel:
        return "betel"
    if tobacco:
        return "tobacco"
    if "none" in blob or "no habit" in blob:
        return "none"
    return "unknown"


def _duration_weeks(ctx: dict[str, Any], notes: str) -> float | None:
    for key in ("lesion_duration_weeks", "duration_weeks", "duration"):
        v = ctx.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    m = re.search(r"duration[_\s]*weeks?\s*[:=]\s*([0-9.]+)", notes, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _map_site_to_canonical(site: str) -> str:
    s = (site or "").lower()
    if "buccal" in s:
        return "buccal mucosa"
    if "tongue" in s:
        return "tongue"
    if "floor" in s:
        return "floor of mouth"
    if "gingiv" in s:
        return "gingiva"
    if "palat" in s:
        return "palate"
    return "unknown"


def _scores_from_b3(probs: np.ndarray, predicted: int) -> dict[str, Any]:
    class_info = CLASSES.get(predicted, CLASSES[0])
    n, o, s = float(probs[0]), float(probs[1]), float(probs[2])
    osmf = min(o, 0.45)  # soft proxy from OPMD channel when no explicit OSMF class
    return {
        "normal": round(n, 4),
        "opmd": round(o, 4),
        "oscc_suspicious": round(s, 4),
        "oscc_confidence": round(s, 4),
        "opmd_confidence": round(o, 4),
        "submucous_fibrosis_confidence": round(osmf, 4),
        "lesion_present": round(1.0 - n, 4),
        "high_risk_habit_confidence": 0.0,
        "biopsy_urgency_confidence": 0.0,
        "_label": class_info["name"],
        "_risk": class_info["risk"],
        "_desc": class_info["description"],
        "_predicted": predicted,
        "_probs": probs,
    }


def _scores_from_vision(v: dict[str, Any]) -> dict[str, Any]:
    def f(key: str, default: float = 0.0) -> float:
        try:
            return float(max(0.0, min(1.0, v.get(key, default))))
        except (TypeError, ValueError):
            return default

    oscc = f("oscc_confidence", 0.1)
    opmd = f("opmd_confidence", 0.1)
    osmf = f("submucous_fibrosis_confidence", 0.05)
    lesion = f("lesion_present", max(oscc, opmd, osmf, 0.2))
    normal = max(0.0, 1.0 - max(oscc, opmd, osmf, lesion * 0.8))
    tot = normal + opmd + oscc + 1e-6
    normal /= tot
    opmd_n = opmd / tot * (1.0 - normal)
    oscc_n = oscc / tot * (1.0 - normal)
    return {
        "normal": round(float(normal), 4),
        "opmd": round(float(opmd_n), 4),
        "oscc_suspicious": round(float(oscc_n), 4),
        "oscc_confidence": round(oscc, 4),
        "opmd_confidence": round(opmd, 4),
        "submucous_fibrosis_confidence": round(osmf, 4),
        "lesion_present": round(lesion, 4),
        "high_risk_habit_confidence": 0.0,
        "biopsy_urgency_confidence": 0.0,
        "_label": str(v.get("lesion_type", "suspicious")),
        "_risk": "high"
        if oscc > 0.55
        else "medium"
        if max(opmd, osmf) > 0.5
        else "low",
        "_desc": str(v.get("narrative", ""))[:400],
        "_vision": v,
    }


def _apply_clinical_rules(
    scores: dict[str, Any],
    ctx: dict[str, Any],
    clinical_notes: str,
    habit_risk: str,
    lesion_location: str,
) -> tuple[dict[str, float], list[str], bool, str]:
    """Returns pathology_scores (floats), emergency_flags, biopsy_recommended, narrative tail."""
    keys = (
        "oscc_confidence",
        "opmd_confidence",
        "submucous_fibrosis_confidence",
        "lesion_present",
        "high_risk_habit_confidence",
        "biopsy_urgency_confidence",
        "normal",
        "opmd",
        "oscc_suspicious",
    )
    ps: dict[str, float] = {}
    for k in keys:
        if k in scores:
            ps[k] = float(scores[k])

    # Habit-driven risk score
    hr = 0.25 if habit_risk in ("tobacco", "betel") else 0.5 if habit_risk == "combined" else 0.1
    if habit_risk == "none":
        hr = 0.05
    ps["high_risk_habit_confidence"] = round(hr, 4)

    flags: list[str] = []
    biopsy = False
    if isinstance(scores.get("_vision"), dict):
        biopsy = bool(scores["_vision"].get("biopsy_recommended", False))

    oc = ps.get("oscc_confidence", ps.get("oscc_suspicious", 0.0))
    op = ps.get("opmd_confidence", ps.get("opmd", 0.0))
    osmf = ps.get("submucous_fibrosis_confidence", 0.0)

    if oc > 0.7:
        flags.append("oral_cancer_suspected")
        biopsy = True
    if op > 0.6 and habit_risk not in ("none", "unknown"):
        flags.append("opmd_high_risk")
    if osmf > 0.45:
        biopsy = True

    dur = _duration_weeks(ctx, clinical_notes)
    if dur is not None and dur > 2.0:
        flags.append("biopsy_required")
        biopsy = True

    # Biopsy urgency scalar
    bu = min(
        1.0,
        0.35 * oc + 0.3 * op + 0.25 * osmf + 0.15 * ps["high_risk_habit_confidence"]
        + (0.25 if "biopsy_required" in flags else 0.0),
    )
    ps["biopsy_urgency_confidence"] = round(float(bu), 4)

    extra = ""
    if dur and dur > 2.0:
        extra += " Lesion duration over 2 weeks — same-week biopsy discussion is appropriate in India practice when suspicious features or habits exist."

    return ps, flags, biopsy, extra


REQUIRED_PATHOLOGY_SCORE_KEYS = (
    "oscc_confidence",
    "opmd_confidence",
    "submucous_fibrosis_confidence",
    "lesion_present",
    "high_risk_habit_confidence",
    "biopsy_urgency_confidence",
    "normal",
    "opmd",
    "oscc_suspicious",
)


def _finalize_pathology_scores(ps: dict[str, Any]) -> dict[str, float]:
    """Ensure correlation + API contract keys exist as floats."""
    out: dict[str, float] = {}
    for k in REQUIRED_PATHOLOGY_SCORE_KEYS:
        v = ps.get(k, 0.0)
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def _canonical_finding_label(raw: str) -> str:
    s = (raw or "").strip()
    if s in ("Normal", "OPMD", "OSCC", "Submucous fibrosis", "Suspicious oral lesion"):
        return s
    low = s.lower().replace("_", " ")
    if low in ("normal", "healthy"):
        return "Normal"
    if "oscc" in low or ("squamous" in low and "carcinoma" in low):
        return "OSCC"
    if "opmd" in low or "leukoplakia" in low or "erythroplakia" in low:
        return "OPMD"
    if "fibrosis" in low or "osmf" in low or "submucous" in low:
        return "Submucous fibrosis"
    if "suspicious" in low:
        return "Suspicious oral lesion"
    return "Suspicious oral lesion"


def _build_findings_from_scores(
    scores: dict[str, Any],
    pathology_scores: dict[str, float],
) -> list[Finding]:
    label = _canonical_finding_label(str(scores.get("_label", "Suspicious oral lesion")))
    risk = scores.get("_risk", "low")
    desc = str(scores.get("_desc", "Automated screening output — clinical correlation required."))
    if label == "Normal":
        sev = "clear"
    elif label == "OSCC" or risk == "high":
        sev = "critical"
    elif label in ("OPMD", "Submucous fibrosis", "Suspicious oral lesion"):
        sev = "warning"
    else:
        sev = "info"
    oc = pathology_scores.get("oscc_confidence", pathology_scores.get("oscc_suspicious", 0.0))
    conf_pct = round(float(oc) * 100.0, 1) if sev == "critical" else round(
        float(pathology_scores.get("lesion_present", 0.3)) * 100.0, 1
    )
    primary = Finding(
        label=label,
        severity=sev,
        confidence=min(100.0, max(0.0, conf_pct)),
        description=desc[:500],
    )
    out: list[Finding] = [primary]
    if pathology_scores.get("submucous_fibrosis_confidence", 0) > 0.35 and "fibrosis" not in label.lower():
        out.append(
            Finding(
                label="Submucous fibrosis",
                severity="warning",
                confidence=round(pathology_scores["submucous_fibrosis_confidence"] * 100.0, 1),
                description="Possible OSMF pattern — assess trismus / areca nut history; biopsy when indicated.",
            )
        )
    return out


def _build_impression(scores: dict[str, Any], pathology_scores: dict[str, float], biopsy: bool) -> str:
    label = str(scores.get("_label", "Lesion"))
    oc = pathology_scores.get("oscc_confidence", 0.0)
    if oc > 0.55:
        return (
            f"High-suspicion oral lesion ({label}) — urgent specialist review and biopsy planning recommended."
        )
    if biopsy:
        return f"Precancerous or suspicious oral findings ({label}) — biopsy recommended; correlate with habits and duration."
    return f"Screening impression: {label}. Routine dental/oral medicine follow-up if symptoms persist."


def _read_oral_cancer_system_prompt(prompt_path: Path | None) -> str:
    p = prompt_path or ORAL_CANCER_SYSTEM_PATH
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    return (
        "You are an oral medicine assistant. Produce structured, India-aware oral screening "
        "reports. AI screening only; tissue diagnosis via biopsy is definitive."
    )


def _openrouter_oral_cancer_narrative_text(
    system: str,
    user_text: str,
    image_b64: str,
) -> tuple[str, str]:
    """OpenRouter narrative (role oral_cancer); vision if image present. Returns (text, model_tag). Never raises."""
    try:
        from llm_router import llm_router
    except Exception:
        return "", ""
    try:
        if image_b64 and len(image_b64.strip()) > 80:
            try:
                raw = base64.b64decode(image_b64)
                mime = _sniff_media_type(raw)
                mime_s = "image/png" if mime == "image/png" else "image/jpeg"
                out = llm_router.complete_for_role(
                    "oral_cancer",
                    system[:200000],
                    user_text[:120000],
                    image_b64=image_b64,
                    image_mime=mime_s,
                    max_tokens=2000,
                )
                txt = (out.get("content") or "").strip()
                if txt:
                    mu = str(out.get("model_used") or "").strip()
                    return txt, mu or "openrouter-oral-narrative"
            except Exception as e:
                logger.warning("OpenRouter oral narrative (vision) failed, trying text: %s", e)
        out2 = llm_router.complete_for_role(
            "oral_cancer",
            system[:200000],
            user_text[:120000],
            max_tokens=2000,
        )
        txt2 = (out2.get("content") or "").strip()
        mu2 = str(out2.get("model_used") or "").strip()
        return txt2, mu2 or "openrouter-oral-narrative"
    except Exception as e:
        logger.warning("OpenRouter oral narrative failed: %s", e)
        return "", ""


def _call_oral_cancer_narrative(
    scores: dict[str, Any],
    patient_context: dict[str, Any],
    structures: dict[str, Any],
    image_b64: str = "",
    prompt_path: Path | None = None,
) -> tuple[str, list[str], str]:
    """
    OpenRouter only (SSOT: config/cloud_inference.yaml role oral_cancer).
    Never raises.
    Returns (narrative_text, additional_emergency_flags, models_used_tag).
    """
    system = _read_oral_cancer_system_prompt(prompt_path)
    ctx = patient_context or {}
    user_text = f"""Oral Cancer Screening Results:

Clinical Impression: {structures.get('lesion_location', 'unknown site')}
Input type: {structures.get('input_type', 'clinical_photo')}

Scores:
  OSCC confidence:              {float(scores.get('oscc_confidence', 0) or 0):.3f}
  OPMD confidence:              {float(scores.get('opmd_confidence', 0) or 0):.3f}
  Submucous fibrosis:           {float(scores.get('submucous_fibrosis_confidence', 0) or 0):.3f}
  Lesion present:               {float(scores.get('lesion_present', 0) or 0):.3f}
  Biopsy urgency confidence:    {float(scores.get('biopsy_urgency_confidence', 0) or 0):.3f}

Lesion site: {structures.get('lesion_location', 'not specified')}
Habit risk:  {structures.get('habit_risk', 'unknown')}
Biopsy recommended: {structures.get('biopsy_recommended', False)}
Emergency flags: {structures.get('emergency_flags', [])}

Patient:
  Age: {ctx.get('age', 'unknown')}
  Sex: {ctx.get('sex', 'unknown')}
  Tobacco habit: {ctx.get('tobacco_habit', 'not specified')}
  Habit duration: {ctx.get('habit_duration_years', 'unknown')} years
  Lesion duration: {ctx.get('lesion_duration_weeks', 'unknown')} weeks
  Clinical history: {ctx.get('clinical_history', 'none provided')}

Generate a structured clinical oral cancer screening report
following the system prompt format.
Include India-specific context if relevant.
Always include biopsy recommendation if any suspicious lesion.
End with the disclaimer that AI screening is not histologic diagnosis."""

    text, mu = _openrouter_oral_cancer_narrative_text(system, user_text, image_b64)
    if text:
        tag = f"OpenRouter-narrative-oral:{mu}" if mu else "OpenRouter-narrative-oral"
        return text, [], tag
    return "", [], ""


def _findings_as_dicts(findings: list[Any]) -> list[dict[str, Any]]:
    """JSON-serializable findings for ZeroClaw / tool responses."""
    out: list[dict[str, Any]] = []
    for f in findings:
        if hasattr(f, "model_dump"):
            out.append(f.model_dump())
        elif isinstance(f, dict):
            out.append(f)
        else:
            out.append(
                {
                    "label": str(f),
                    "severity": "info",
                    "confidence": 0.0,
                    "description": "",
                }
            )
    return out


def _map_image_type_to_input(s: str) -> str | None:
    """Map ZeroClaw image_type enum → pipeline input_type."""
    x = (s or "").strip().lower()
    if x in ("clinical_photo", "intraoral"):
        return "clinical_photo"
    if x == "histopathology":
        return "histopathology"
    if x == "unknown":
        return "unknown"
    if x in ("mixed",):
        return "mixed"
    return None


def run_oral_cancer_pipeline(
    filepath: str,
    job_id: str,
    clinical_notes: str = "",
    patient_context: dict[str, Any] | None = None,
    input_type_override: str | None = None,
) -> dict[str, Any]:
    if not ORAL_CANCER_ENABLED:
        raise OralServiceUnavailableError("Oral cancer service is disabled (ORAL_CANCER_ENABLED=false).")

    ctx = _merge_patient_context(clinical_notes, patient_context)
    try:
        pil_img = PILImage.open(filepath).convert("RGB")
    except Exception as e:
        logger.exception("Image load failed: %s", e)
        return _minimal_safe_response(job_id, clinical_notes, ctx, str(e))

    with open(filepath, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    input_type = _infer_input_type(filepath, ctx, input_type_override)
    habit_risk = _habit_risk_from_context(ctx, clinical_notes)
    lesion_location = _map_site_to_canonical(str(ctx.get("location_body", ctx.get("lesion_site", ""))))

    models_used: list[str] = []
    limitation_notes: list[str] = []
    score_bundle: dict[str, Any] | None = None
    model_path_note = ""

    # Path B: histopathology-first
    if input_type in ("histopathology", "mixed"):
        uni = _uni_embedding_and_scores(pil_img)
        if uni:
            probs = uni["probs3"]
            pred = uni["predicted"]
            score_bundle = _scores_from_b3(np.asarray(probs), pred)
            if uni.get("limitation"):
                limitation_notes.append(uni["limitation"])
            model_path_note = uni.get("model_path") or UNI_MODEL_ID
            models_used.append("UNI-pathology-encoder")
            if uni.get("model_path"):
                models_used.append("UNI-linear-head")

    # Path A: clinical photo — B3 then V2M
    if score_bundle is None and input_type in ("clinical_photo", "mixed", "unknown"):
        b3 = _run_b3_classification(pil_img)
        if b3 is not None:
            probs, pred = b3
            score_bundle = _scores_from_b3(probs, pred)
            models_used.append("EfficientNet-B3")
            model_path_note = _checkpoint_path()
        else:
            v2 = _run_v2m_classification(pil_img)
            if v2 is not None:
                probs, pred = v2
                score_bundle = _scores_from_b3(probs, pred)
                models_used.append("EfficientNet-V2-M")
                model_path_note = _v2m_path()

    # If histopathology but UNI failed, try clinical models as weak second
    if score_bundle is None and input_type == "histopathology":
        b3 = _run_b3_classification(pil_img)
        if b3 is not None:
            score_bundle = _scores_from_b3(*b3)
            models_used.append("EfficientNet-B3-fallback")
            limitation_notes.append("Clinical classifier applied to non-clinical image — interpret with caution.")

    vision_data: dict[str, Any] | None = None
    vision_provider = ""
    vision_model_slug = ""
    if score_bundle is None:
        vision_data, vision_provider, vision_model_slug = _vision_llm_oral(
            pil_img, image_b64, ctx, clinical_notes
        )
        if vision_data:
            score_bundle = _scores_from_vision(vision_data)
            if vision_provider == "openrouter":
                models_used.append("openrouter-vision-oral")
                if vision_model_slug:
                    models_used.append(vision_model_slug)
            lesion_location = _map_site_to_canonical(str(vision_data.get("lesion_site", lesion_location)))
            habit_v = str(vision_data.get("habit_risk", ""))
            if habit_v in ("tobacco", "betel", "combined", "none", "unknown"):
                habit_risk = habit_v
            model_path_note = "openrouter-vision"
        else:
            score_bundle = {
                "normal": 0.34,
                "opmd": 0.33,
                "oscc_suspicious": 0.33,
                "oscc_confidence": 0.33,
                "opmd_confidence": 0.33,
                "submucous_fibrosis_confidence": 0.2,
                "lesion_present": 0.4,
                "high_risk_habit_confidence": 0.0,
                "biopsy_urgency_confidence": 0.25,
                "_label": "Suspicious oral lesion",
                "_risk": "medium",
                "_desc": "No local classifier weights and vision LLM unavailable — non-specific placeholder scores only.",
            }
            limitation_notes.append(
                "No fine-tuned oral weights and no OpenRouter vision fallback — output is intentionally non-specific."
            )
            models_used.append("limitation-only")

    pathology_scores, emergency_flags, biopsy_rec, rule_tail = _apply_clinical_rules(
        score_bundle, ctx, clinical_notes, habit_risk, lesion_location
    )

    # Merge legacy + extended keys (all floats)
    for k in ("normal", "opmd", "oscc_suspicious"):
        if k in score_bundle:
            pathology_scores[k] = round(float(score_bundle[k]), 4)

    pathology_scores = _finalize_pathology_scores(pathology_scores)

    findings = _build_findings_from_scores(score_bundle, pathology_scores)
    impression = _build_impression(score_bundle, pathology_scores, biopsy_rec) + rule_tail

    narrative = ""
    india_note = ""
    if isinstance(score_bundle.get("_vision"), dict):
        narrative = str(score_bundle["_vision"].get("narrative", ""))
        india_note = str(score_bundle["_vision"].get("india_note", ""))
    if not narrative:
        narrative = impression
    if limitation_notes:
        india_note = (india_note + " " + " ".join(limitation_notes)).strip()

    conf_val = float(pathology_scores.get("oscc_confidence", 0.0))
    confidence = "high" if conf_val > 0.65 else "medium" if conf_val > 0.35 else "low"

    structures: dict[str, Any] = {
        "lesion_location": lesion_location,
        "habit_risk": habit_risk,
        "biopsy_recommended": biopsy_rec,
        "emergency_flags": emergency_flags,
        "narrative_report": narrative,
        "india_note": india_note.strip(),
        "model_path": model_path_note,
        "input_type": input_type,
        "clinical_notes": clinical_notes or "",
        "patient_context": ctx,
        "predicted_class": str(score_bundle.get("_label", "")),
        "checkpoint_used": _has_b3_checkpoint(),
        "limitations": limitation_notes,
    }

    narr_llm, extra_narr_flags, narr_tag = _call_oral_cancer_narrative(
        scores=pathology_scores,
        patient_context=ctx,
        structures=structures,
        image_b64=image_b64,
        prompt_path=ORAL_CANCER_SYSTEM_PATH,
    )
    if narr_llm:
        structures["narrative_report"] = narr_llm
        if extra_narr_flags:
            merged_ef = list(structures.get("emergency_flags", [])) + list(extra_narr_flags)
            structures["emergency_flags"] = list(dict.fromkeys(merged_ef))
        if narr_tag:
            models_used.append(narr_tag)

    return {
        "modality": "oral_cancer",
        "findings": _findings_as_dicts(findings),
        "impression": impression[:800],
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": confidence,
        "confidence_score": conf_val,
        "models_used": models_used,
        "disclaimer": f"{DISCLAIMER} {ORAL_DISCLAIMER}".strip(),
    }


def oral_degraded_from_exception(
    job_id: str,
    clinical_notes: str,
    patient_context: dict[str, Any] | None,
    err: str,
) -> dict[str, Any]:
    """Public entry for HTTP layer when the pipeline raises unexpectedly."""
    ctx = _merge_patient_context(clinical_notes, patient_context)
    return _minimal_safe_response(job_id, clinical_notes, ctx, err)


def _minimal_safe_response(
    job_id: str,
    clinical_notes: str,
    ctx: dict[str, Any],
    err: str,
) -> dict[str, Any]:
    ps = {
        "normal": 0.34,
        "opmd": 0.33,
        "oscc_suspicious": 0.33,
        "oscc_confidence": 0.33,
        "opmd_confidence": 0.33,
        "submucous_fibrosis_confidence": 0.2,
        "lesion_present": 0.0,
        "high_risk_habit_confidence": 0.1,
        "biopsy_urgency_confidence": 0.2,
    }
    ps = _finalize_pathology_scores(ps)
    return {
        "modality": "oral_cancer",
        "findings": _findings_as_dicts(
            [
                Finding(
                    label="Suspicious oral lesion",
                    severity="info",
                    confidence=30.0,
                    description=f"Image could not be processed ({err}). Clinical examination required.",
                )
            ]
        ),
        "impression": "Analysis unavailable — please re-capture image or examine clinically.",
        "pathology_scores": ps,
        "structures": {
            "lesion_location": "unknown",
            "habit_risk": _habit_risk_from_context(ctx, clinical_notes),
            "biopsy_recommended": False,
            "emergency_flags": [],
            "narrative_report": "",
            "india_note": "Upload a clear intraoral photo or histopathology crop. Persistent lesions with tobacco/areca exposure need biopsy discussion.",
            "model_path": "",
            "input_type": "unknown",
            "clinical_notes": clinical_notes,
            "patient_context": ctx,
            "error": err,
        },
        "confidence": "low",
        "confidence_score": 0.2,
        "models_used": [],
        "disclaimer": f"{DISCLAIMER} {ORAL_DISCLAIMER}".strip(),
    }


def run_oral_cancer_pipeline_b64(
    image_b64: str,
    clinical_notes: str = "",
    job_id: str = "",
    filename_hint: str = "",
    patient_context: dict[str, Any] | None = None,
    patient_context_json: str | None = None,
    input_type: str | None = None,
    image_type: str | None = None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = dict(patient_context or {})
    if patient_context_json and str(patient_context_json).strip():
        try:
            parsed = json.loads(patient_context_json)
            if isinstance(parsed, dict):
                ctx = {**ctx, **parsed}
        except json.JSONDecodeError:
            logger.warning("Invalid patient_context_json in oral b64 pipeline; ignoring.")

    override = input_type
    if image_type and str(image_type).strip():
        mapped = _map_image_type_to_input(str(image_type))
        if mapped:
            override = mapped

    try:
        data = base64.b64decode(image_b64, validate=False)
    except Exception as e:
        logger.warning("Invalid base64 for oral pipeline: %s", e)
        return oral_degraded_from_exception(
            job_id or "zeroclaw",
            clinical_notes,
            ctx,
            f"invalid base64: {e}",
        )
    ext = os.path.splitext(filename_hint)[1].lower() if filename_hint else ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        ext = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data)
        fp = f.name
    try:
        return run_oral_cancer_pipeline(
            fp,
            job_id or "zeroclaw",
            clinical_notes=clinical_notes,
            patient_context=ctx,
            input_type_override=override,
        )
    finally:
        try:
            os.unlink(fp)
        except OSError:
            pass
