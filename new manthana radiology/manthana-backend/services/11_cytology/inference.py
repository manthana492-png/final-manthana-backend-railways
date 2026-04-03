"""Manthana — Cytology Inference: Virchow + DSMIL + Kimi / Claude narrative."""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# ── Shared path resolver ────────────────────────────────


def _find_shared() -> Path:
    """Resolve the shared/ directory (Docker, Lightning AI, local repo)."""
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
    raise RuntimeError(
        "Cannot find shared/ directory. Searched: " + ", ".join(str(c) for c in candidates)
    )


_shared = _find_shared()
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from env_file import load_api_keys_env

load_api_keys_env()

from dsmil_aggregator import dsmil_slide_scores
from disclaimer import DISCLAIMER
from model_loader import LazyModel
from tile_embedding import run_tile_embeddings

logger = logging.getLogger("manthana.cytology")

virchow_model = LazyModel(model_id="paige-ai/Virchow", cache_name="virchow", device="cuda")

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "cytology_system.txt"
CLAUDE_MODEL = os.environ.get("CLAUDE_CYTOLOGY_MODEL", "claude-sonnet-4-20250514")

BETHESDA_LABELS = ["NILM", "ASCUS", "LSIL", "HSIL", "AGC", "AIS", "malignant"]
SPUTUM_LABELS = ["normal", "reactive_changes", "suspicious", "malignant", "tb_suggestive"]
FNAC_LABELS = ["benign", "atypical", "suspicious_malignancy", "malignant"]
URINE_LABELS = ["negative", "atypical", "suspicious", "high_grade", "malignant"]
CSF_LABELS = ["no_malignant_cells", "atypical_cells", "malignant_cells"]

CRITICAL_CLASSES: dict[str, set[str]] = {
    "pap_smear": {"HSIL", "AIS", "malignant", "AGC"},
    "sputum": {"suspicious", "malignant"},
    "fnac": {"suspicious_malignancy", "malignant"},
    "urine": {"high_grade", "malignant"},
    "csf": {"malignant_cells"},
    "ascitic": {"malignant", "suspicious"},
    "other": {"malignant"},
}


def _label_set(specimen_type: str) -> list[str]:
    return {
        "pap_smear": BETHESDA_LABELS,
        "sputum": SPUTUM_LABELS,
        "fnac": FNAC_LABELS,
        "urine": URINE_LABELS,
        "csf": CSF_LABELS,
        "ascitic": FNAC_LABELS,
    }.get(specimen_type, FNAC_LABELS)


def is_loaded() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except Exception:
        return virchow_model.is_loaded()


def _safe_scores(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def _assign_label_scores(labels: list[str], mal: float, ben: float) -> dict[str, float]:
    scores = {lab: 0.0 for lab in labels}
    benign_candidates = ("NILM", "negative", "normal", "benign", "no_malignant_cells", "reactive_changes")
    malignant_candidates = (
        "malignant",
        "malignant_cells",
        "high_grade",
        "HSIL",
        "AIS",
        "suspicious_malignancy",
        "suspicious",
    )
    benign_lab = next((l for l in labels if l in benign_candidates), labels[0])
    malignant_lab = next((l for l in labels if l in malignant_candidates), labels[-1])
    scores[benign_lab] = ben
    scores[malignant_lab] = mal
    others = [l for l in labels if l not in (benign_lab, malignant_lab)]
    rem = max(0.0, 1.0 - mal - ben)
    if others:
        per = rem / len(others)
        for l in others:
            scores[l] = per
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 4) for k, v in scores.items()}
    return scores


def _analyze_cells(embeddings_list: list, specimen_type: str) -> dict[str, Any]:
    labels = _label_set(specimen_type)
    n_cells = len(embeddings_list)
    if n_cells == 0:
        return {
            "classification_status": "no_embeddings",
            "top_class": labels[0],
            "confidence": 0.0,
            "is_critical": False,
            "specimen_type": specimen_type,
            "adequacy": "Unsatisfactory — no cells extracted",
            "cell_count": 0,
        }

    raw = dsmil_slide_scores(embeddings_list, num_classes=4)
    mal = float(raw.get("malignancy_score") or 0.0)
    ben = float(raw.get("benign_score") or 0.0)
    infl = float(raw.get("inflammation_score") or 0.0)
    necr = float(raw.get("necrosis_score") or 0.0)

    scores = _assign_label_scores(labels, mal, ben)
    top_class = max(scores, key=scores.get)
    top_score = float(scores.get(top_class, 0.0))

    crit_set = CRITICAL_CLASSES.get(specimen_type, CRITICAL_CLASSES["other"])
    is_critical = (mal >= 0.55) or (top_class in crit_set and top_score >= 0.45)

    if specimen_type == "pap_smear":
        adequacy = (
            "Satisfactory for evaluation"
            if n_cells >= 20
            else "Unsatisfactory — insufficient cellularity for AI screening"
        )
    else:
        adequacy = "Adequate" if n_cells >= 5 else "Limited — low cellularity"

    out: dict[str, Any] = {
        "classification_status": str(raw.get("classification_status", "dsmil_mil")),
        "top_class": top_class,
        "confidence": round(top_score, 4),
        "is_critical": is_critical,
        "specimen_type": specimen_type,
        "adequacy": adequacy,
        "cell_count": n_cells,
        "malignant": mal,
        "tb_suggestive": min(1.0, infl * 0.6 + necr * 0.25) if specimen_type == "sputum" else 0.0,
        "is_critical_flag": 1.0 if is_critical else 0.0,
    }
    if specimen_type == "pap_smear":
        out["bethesda_category"] = top_class
    out.update(scores)
    out["is_critical"] = is_critical
    return out


def _build_cyto_findings(cell_result: dict[str, Any], specimen_type: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    is_critical = bool(cell_result.get("is_critical"))
    top = str(cell_result.get("top_class", ""))
    conf = float(cell_result.get("confidence", 0.0))
    adequacy = str(cell_result.get("adequacy", ""))

    if is_critical:
        label_map = {
            "pap_smear": "Urgent — high-grade cytology pattern (HSIL/AIS/malignant) — colposcopy referral",
            "sputum": "Urgent — suspicious or malignant cells in sputum",
            "fnac": "Urgent — suspicious or malignant FNAC pattern",
            "urine": "Urgent — high-grade or malignant urothelial cells",
            "csf": "Urgent — malignant cells in CSF",
        }
        findings.append(
            {
                "label": label_map.get(specimen_type, "Urgent — malignant pattern on cytology"),
                "severity": "critical",
                "confidence": min(100.0, conf * 100.0),
                "description": f"Top class: {top}",
            }
        )
    else:
        findings.append(
            {
                "label": top.replace("_", " ").title(),
                "severity": "warning" if conf > 0.50 else "info",
                "confidence": min(100.0, conf * 100.0),
                "description": f"Specimen adequacy: {adequacy}",
            }
        )

    if "Unsatisfactory" in adequacy or "Limited" in adequacy:
        findings.append(
            {
                "label": "Specimen adequacy concern",
                "severity": "warning",
                "confidence": 0.0,
                "description": adequacy,
            }
        )
    return findings


def _extract_cells(filepath: str):
    try:
        from preprocessing.wsi_utils import extract_tiles

        return extract_tiles(filepath, tile_size=128, tissue_threshold=0.3)
    except Exception:
        from preprocessing.image_utils import load_image

        return [(load_image(filepath), {"x": 0, "y": 0})]


def _sniff_media_type(image_bytes: bytes) -> str:
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"


def _call_claude_narrative_only(
    cell_result: dict[str, Any],
    structures: dict[str, Any],
    patient_context: dict[str, Any],
    image_b64: str,
    claude_client: Any,
) -> str:
    if claude_client is None:
        return ""
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Prompt read failed: %s", e)
        return ""

    measurements = json.dumps(
        {
            "cytology_scores": {k: v for k, v in cell_result.items()
                                if k in ("top_class", "confidence", "adequacy", "specimen_type", "bethesda_category")},
            "patient_context": patient_context,
        },
        indent=2,
        ensure_ascii=False,
    )

    try:
        msg = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _sniff_media_type(base64.b64decode(image_b64)),
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "QUANTITATIVE MEASUREMENTS:\n"
                                f"{measurements}\n\n"
                                "Generate a structured cytopathology-style narrative for clinician review."
                            ),
                        },
                    ],
                }
            ],
        )
        return msg.content[0].text
    except Exception as e:
        logger.warning("Claude narrative failed: %s", e)
        return f"[Narrative unavailable: {e}]"


def _call_kimi_cytology_narrative(
    cell_result: dict[str, Any],
    structures: dict[str, Any],
    patient_context: dict[str, Any],
    scores: dict[str, float],
) -> str:
    key = (os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or "").strip()
    if not key:
        return ""
    try:
        from openai import OpenAI
    except ImportError:
        return ""

    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        system_prompt = "You are an expert cytopathology assistant."

    model = (
        os.environ.get("KIMI_CYTOLOGY_MODEL", "").strip()
        or os.environ.get("KIMI_MODEL", "moonshot-v1-8k").strip()
    )
    base = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1").strip()
    age = patient_context.get("age", "")
    sex = patient_context.get("sex", "")
    clin = patient_context.get("clinical_history", patient_context.get("history", ""))

    user_text = f"""Cytology Analysis Results:
Specimen type: {structures.get('specimen_type')}
Bethesda category: {structures.get('bethesda_category')}
Adequacy: {structures.get('adequacy')}
Cell count: {structures.get('cell_count', 0)}
Atypical cell fraction: {structures.get('atypical_cell_fraction', 0):.3f}

Cytology scores:
  HSIL confidence: {scores.get('hsil_confidence', 0):.3f}
  N/C ratio: {scores.get('n_c_ratio', 0):.3f}
  Nuclear irregularity: {scores.get('nuclear_irregularity', 0):.3f}

Patient: {age}y {sex}
HPV status: {patient_context.get('hpv_status', 'unknown')}
History: {clin}

Generate a cytopathology report following the system prompt format."""

    try:
        client = OpenAI(api_key=key, base_url=base)
        r = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Kimi cytology narrative failed: %s", e)
        return ""


def _cytology_narrative_kimi_then_claude(
    cell_result: dict[str, Any],
    structures: dict[str, Any],
    patient_context: dict[str, Any],
    pathology_scores: dict[str, float],
    image_b64: str,
    claude_client: Any,
) -> tuple[str, list[str]]:
    tags: list[str] = []
    kimi_text = _call_kimi_cytology_narrative(
        cell_result, structures, patient_context, pathology_scores
    )
    if kimi_text:
        tags.append("Kimi-narrative-Cytology")
        return kimi_text, tags

    claude_text = _call_claude_narrative_only(
        cell_result, structures, patient_context, image_b64, claude_client
    )
    if claude_text and not claude_text.startswith("[Narrative unavailable"):
        tags.append("Claude-narrative-Cytology")
    return claude_text, tags


def run_pipeline(
    filepath: str,
    job_id: str = "",
    patient_context: Optional[dict[str, Any]] = None,
    claude_client: Any = None,
    image_b64: str = "",
) -> dict[str, Any]:
    if patient_context is None:
        patient_context = {}

    specimen_type = str(patient_context.get("specimen_type") or "fnac").lower()
    if specimen_type not in {"pap_smear", "sputum", "fnac", "urine", "csf", "ascitic", "other"}:
        specimen_type = "fnac"

    logger.info("[%s] Running cytology pipeline (specimen=%s)...", job_id or "local", specimen_type)

    tiles = _extract_cells(filepath)
    embeddings, emb_name, lim_note = run_tile_embeddings(tiles, virchow_model, max_tiles=100)
    cell_result = _analyze_cells(embeddings, specimen_type)

    findings = _build_cyto_findings(cell_result, specimen_type)

    labels = _label_set(specimen_type)
    score_dict: dict[str, Any] = {k: v for k, v in cell_result.items() if k in labels}
    score_dict["malignant"] = float(cell_result.get("malignant", 0.0))
    score_dict["tb_suggestive"] = float(cell_result.get("tb_suggestive", 0.0))
    score_dict["is_critical"] = 1.0 if cell_result.get("is_critical") else 0.0
    mal = float(cell_result.get("malignant", 0.0))
    hsil_c = float(cell_result.get("HSIL", 0.0)) if specimen_type == "pap_smear" else 0.0
    pathology_scores = _safe_scores(score_dict)
    pathology_scores["hsil_confidence"] = hsil_c
    pathology_scores["n_c_ratio"] = min(1.0, mal * 0.85 + 0.05)
    pathology_scores["nuclear_irregularity"] = min(1.0, mal * 0.9)

    atypical_frac = 1.0 - float(cell_result.get("NILM", 0.0)) if specimen_type == "pap_smear" else mal

    structures: dict[str, Any] = {
        "specimen_type": specimen_type,
        "bethesda_category": cell_result.get("bethesda_category", ""),
        "adequacy": cell_result.get("adequacy", ""),
        "is_critical": bool(cell_result.get("is_critical")),
        "cell_count": cell_result.get("cell_count", 0),
        "atypical_cell_fraction": round(float(atypical_frac), 4),
        "model_name": emb_name,
        "patient_context": patient_context,
        "narrative_report": "",
    }
    if lim_note:
        structures["limitation_note"] = lim_note

    if not image_b64:
        try:
            with open(filepath, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("ascii")
        except OSError:
            image_b64 = ""

    if claude_client is None and os.environ.get("ANTHROPIC_API_KEY") and image_b64:
        try:
            import anthropic

            claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        except Exception as e:
            logger.debug("Anthropic client not created: %s", e)

    narrative, narr_tags = _cytology_narrative_kimi_then_claude(
        cell_result=cell_result,
        structures=structures,
        patient_context=patient_context,
        pathology_scores=pathology_scores,
        image_b64=image_b64,
        claude_client=claude_client,
    )
    structures["narrative_report"] = narrative

    models_used: list[str] = [emb_name, "DSMIL-MIL"] + narr_tags

    if narrative and not narrative.startswith("[Narrative unavailable"):
        impression = narrative[:500].strip()
    else:
        impression = (
            f"Top class {cell_result.get('top_class')}. "
            f"Adequacy: {cell_result.get('adequacy', 'unknown')}."
        )

    emergency_flags: list[str] = []
    if specimen_type == "pap_smear" and (
        float(pathology_scores.get("hsil_confidence", 0)) > 0.5
        or str(structures.get("bethesda_category", "")).upper() == "HSIL"
    ):
        emergency_flags.extend(["colposcopy", "HSIL", "refer", "biopsy"])
    structures["emergency_flags"] = emergency_flags

    return {
        "modality": "cytology",
        "findings": findings,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": "medium",
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }


def run_cytology_pipeline_b64(
    image_b64: str,
    patient_context: Optional[dict[str, Any]] = None,
    claude_client: Any = None,
) -> dict[str, Any]:
    import tempfile

    if patient_context is None:
        patient_context = {}
    raw = base64.b64decode(image_b64)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        return run_pipeline(
            filepath=tmp,
            job_id="zeroclaw",
            patient_context=patient_context,
            claude_client=claude_client,
            image_b64=image_b64,
        )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
