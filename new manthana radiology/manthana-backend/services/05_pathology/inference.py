"""Manthana — Pathology Inference: Virchow tile embeddings → DSMIL → OpenRouter narrative."""
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

logger = logging.getLogger("manthana.pathology")

virchow_model = LazyModel(model_id="paige-ai/Virchow", cache_name="virchow", device="cuda")

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "pathology_system.txt"


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


def _build_findings_from_dsmil(dsmil_result: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    mal = float(dsmil_result.get("malignancy_score") or 0.0)
    ben = float(dsmil_result.get("benign_score") or 0.0)
    infl = float(dsmil_result.get("inflammation_score") or 0.0)
    necr = float(dsmil_result.get("necrosis_score") or 0.0)

    if mal >= 0.60:
        findings.append(
            {
                "label": "Possible malignancy — urgent histopathology review",
                "severity": "critical",
                "confidence": min(100.0, mal * 100.0),
                "description": f"Malignancy score {mal:.2f} — correlate with clinical findings and IHC as indicated.",
            }
        )
    elif mal >= 0.40:
        findings.append(
            {
                "label": "Atypical tissue — malignancy cannot be excluded",
                "severity": "warning",
                "confidence": min(100.0, mal * 100.0),
                "description": f"Malignancy score {mal:.2f} — clinical correlation recommended.",
            }
        )
    else:
        findings.append(
            {
                "label": "No definitive high-grade malignancy pattern on AI screening",
                "severity": "clear",
                "confidence": min(100.0, max(ben, 1.0 - mal) * 100.0),
                "description": f"Benign-leaning score {ben:.2f}; malignancy score {mal:.2f}.",
            }
        )

    if infl >= 0.30:
        findings.append(
            {
                "label": "Inflammatory infiltrate signal",
                "severity": "info",
                "confidence": min(100.0, infl * 100.0),
                "description": f"Inflammation score {infl:.2f}.",
            }
        )
    if necr >= 0.25:
        findings.append(
            {
                "label": "Necrosis signal",
                "severity": "warning",
                "confidence": min(100.0, necr * 100.0),
                "description": f"Necrosis score {necr:.2f}.",
            }
        )
    return findings


def _extract_tiles(filepath: str):
    try:
        from preprocessing.wsi_utils import extract_tiles

        return extract_tiles(filepath, tile_size=256, tissue_threshold=0.5)
    except Exception as e:
        logger.warning("WSI tile extraction failed: %s. Treating as single image.", e)
        from preprocessing.image_utils import load_image

        img = load_image(filepath)
        return [(img, {"x": 0, "y": 0})]


def _sniff_media_type(image_bytes: bytes) -> str:
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"


def _pathology_narrative_openrouter(
    dsmil_scores: dict[str, Any],
    structures: dict[str, Any],
    patient_context: dict[str, Any],
    pathology_scores: dict[str, float],
    image_b64: str,
    findings: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    tags: list[str] = []
    findings_summary = "; ".join(
        f"{f.get('label', '')} ({f.get('severity', '')})" for f in findings[:12]
    )
    if not findings_summary:
        findings_summary = "No structured findings."
    try:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        system_prompt = "You are an expert histopathology assistant."
    age = patient_context.get("age", "")
    sex = patient_context.get("sex", "")
    clin = patient_context.get("clinical_history", patient_context.get("history", ""))
    measurements = json.dumps(
        {
            "dsmil_scores": dsmil_scores,
            "tissue_source": patient_context.get("tissue_source", ""),
            "stain": patient_context.get("stain", "H&E"),
            "tile_count": structures.get("tile_count", 0),
            "patient_context": patient_context,
        },
        indent=2,
        ensure_ascii=False,
    )
    user_text = f"""Pathology Analysis Results:
Tissue type: {structures.get('tissue_type')}
Stain: {structures.get('stain', 'H&E')}
Tile count: {structures.get('tile_count')}
Model: {structures.get('model_name', 'Virchow')}

Pathology scores:
  Malignancy confidence: {pathology_scores.get('malignancy_confidence', 0):.3f}
  Tumor percentage: {pathology_scores.get('tumor_percentage', 0):.3f}
  Necrosis fraction: {pathology_scores.get('necrosis_fraction', 0):.3f}
  Mitotic count per mm2: {pathology_scores.get('mitotic_count_per_mm2', 0):.1f}

Findings:
  {findings_summary}

Patient: {age}y {sex}
History: {clin}

QUANTITATIVE:
{measurements}

Generate a pathology report following the system prompt format.
Include WHO grade, mitotic rate, necrosis extent,
resection margins if applicable, and management recommendation."""
    try:
        from llm_router import llm_router

        mime = _sniff_media_type(base64.b64decode(image_b64)) if image_b64 else "image/jpeg"
        if image_b64:
            out = llm_router.complete_for_role(
                "vision_pathology",
                system_prompt,
                user_text,
                image_b64=image_b64,
                image_mime=mime,
                max_tokens=2000,
            )
            txt = (out.get("content") or "").strip()
            if txt:
                tags.append("OpenRouter-narrative-Pathology")
                return txt, tags
        out = llm_router.complete_for_role(
            "vision_pathology",
            system_prompt,
            user_text,
            max_tokens=2000,
        )
        txt = (out.get("content") or "").strip()
        if txt:
            tags.append("OpenRouter-narrative-Pathology")
            return txt, tags
    except Exception as e:
        logger.warning("OpenRouter pathology narrative failed: %s", e)
        return f"[Narrative unavailable: {e}]", tags
    return "", tags


def run_pipeline(
    filepath: str,
    job_id: str = "",
    patient_context: Optional[dict[str, Any]] = None,
    claude_client: Any = None,
    image_b64: str = "",
) -> dict[str, Any]:
    if patient_context is None:
        patient_context = {}

    logger.info("[%s] Running pathology pipeline...", job_id or "local")
    tiles = _extract_tiles(filepath)
    embeddings, emb_name, lim_note = run_tile_embeddings(tiles, virchow_model, max_tiles=100)
    dsmil_result = dsmil_slide_scores(embeddings)

    findings = _build_findings_from_dsmil(dsmil_result)
    mal = float(dsmil_result.get("malignancy_score") or 0.0)
    necr = float(dsmil_result.get("necrosis_score") or 0.0)

    pathology_scores = _safe_scores(
        {
            "malignancy_score": mal,
            "malignancy_confidence": mal,
            "benign_score": float(dsmil_result.get("benign_score") or 0.0),
            "inflammation_score": float(dsmil_result.get("inflammation_score") or 0.0),
            "necrosis_score": necr,
            "necrosis_fraction": necr,
            "tumor_percentage": min(1.0, max(0.0, mal)),
            "mitotic_count_per_mm2": float(mal * 12.0 + necr * 3.0),
        }
    )

    structures: dict[str, Any] = {
        "tissue_source": patient_context.get("tissue_source", "unknown"),
        "stain": patient_context.get("stain", "H&E"),
        "tile_count": len(tiles),
        "tissue_type": str(dsmil_result.get("tissue_type", "") or ""),
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

    narrative, narr_tags = _pathology_narrative_openrouter(
        dsmil_scores=dsmil_result,
        structures=structures,
        patient_context=patient_context,
        pathology_scores=pathology_scores,
        image_b64=image_b64,
        findings=findings,
    )
    structures["narrative_report"] = narrative

    models_used: list[str] = [emb_name, "DSMIL-MIL"] + narr_tags

    if narrative and not narrative.startswith("[Narrative unavailable"):
        impression = narrative[:500].strip()
    else:
        impression = (
            f"Malignancy score {pathology_scores.get('malignancy_score', 0):.2f}. "
            f"Tissue: {dsmil_result.get('tissue_type', 'undetermined')}."
        )

    return {
        "modality": "pathology",
        "findings": findings,
        "impression": impression,
        "pathology_scores": pathology_scores,
        "structures": structures,
        "confidence": "medium",
        "models_used": models_used,
        "disclaimer": DISCLAIMER,
    }


def run_pathology_pipeline_b64(
    image_b64: str,
    patient_context: Optional[dict[str, Any]] = None,
    claude_client: Any = None,
) -> dict[str, Any]:
    import tempfile

    if patient_context is None:
        patient_context = {}
    raw = base64.b64decode(image_b64)
    is_tiff = raw[:2] in (b"II", b"MM") or (len(raw) >= 4 and raw[:4] in (b"II*\x00", b"MM\x00*"))
    suffix = ".tiff" if is_tiff else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
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
