"""
Manthana — Bone / extremity / spine / skull X-Ray pipeline (MSK).

Deterministic image-derived proxy scoring with normalized response schema.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from disclaimer import DISCLAIMER

logger = logging.getLogger("manthana.xray.bone")


def _load_gray(filepath: str) -> np.ndarray:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read X-ray file: {filepath}")
    return img.astype(np.float32) / 255.0


def _bone_scores_from_gray(g: np.ndarray, region: str) -> dict[str, float]:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(grad > np.percentile(grad, 75)))
    contrast = float(np.std(g))
    dark_ratio = float(np.mean(g < 0.25))
    bright_ratio = float(np.mean(g > 0.75))

    line_img = (grad / (float(np.max(grad)) + 1e-6) * 255.0).astype(np.uint8)
    lines = cv2.HoughLinesP(
        line_img,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=max(10, int(min(g.shape) * 0.04)),
        maxLineGap=8,
    )
    line_count = float(0 if lines is None else len(lines))
    size_norm = max(float(g.shape[0] * g.shape[1]) / 1e5, 1.0)
    line_density = min(1.0, line_count / (30.0 * size_norm))

    fracture_proxy = float(np.clip(0.55 * line_density + 0.45 * edge_density, 0.0, 1.0))
    osteopenia_proxy = float(np.clip(0.7 * dark_ratio + 0.3 * (1.0 - contrast), 0.0, 1.0))
    degeneration_proxy = float(np.clip(0.5 * edge_density + 0.5 * bright_ratio, 0.0, 1.0))

    out = {
        "fracture_proxy": fracture_proxy,
        "osteopenia_proxy": osteopenia_proxy,
        "degenerative_change_proxy": degeneration_proxy,
        "edge_density": edge_density,
        "contrast_score": contrast,
        "line_density": line_density,
    }
    if region == "spine":
        out["disc_space_narrowing_proxy"] = float(
            np.clip(0.6 * degeneration_proxy + 0.4 * bright_ratio, 0.0, 1.0)
        )
    return out


def _findings(scores: dict[str, float], region: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if scores.get("fracture_proxy", 0.0) > 0.6:
        out.append(
            {
                "label": "Possible fracture pattern",
                "severity": "warning",
                "confidence": round(scores["fracture_proxy"] * 100.0, 1),
                "region": region.title(),
                "description": "Linear cortical discontinuity proxy elevated on edge-line analysis.",
            }
        )
    if scores.get("osteopenia_proxy", 0.0) > 0.55:
        out.append(
            {
                "label": "Low mineralization pattern",
                "severity": "info",
                "confidence": round(scores["osteopenia_proxy"] * 100.0, 1),
                "region": region.title(),
                "description": "Diffuse low-density pattern suggests reduced bone mineralization.",
            }
        )
    if region == "spine" and scores.get("disc_space_narrowing_proxy", 0.0) > 0.55:
        out.append(
            {
                "label": "Possible spondylotic change",
                "severity": "info",
                "confidence": round(scores["disc_space_narrowing_proxy"] * 100.0, 1),
                "region": "Spine",
                "description": "Disc-space narrowing proxy elevated; correlate with clinical symptoms.",
            }
        )
    if not out:
        out.append(
            {
                "label": "No high-risk MSK signal",
                "severity": "clear",
                "confidence": 88.0,
                "region": region.title(),
                "description": "No major abnormality flagged by deterministic MSK feature analysis.",
            }
        )
    return out


def _impression(scores: dict[str, float], region: str) -> str:
    if scores.get("fracture_proxy", 0.0) > 0.65:
        return f"{region.title()} radiograph shows elevated fracture proxy; urgent radiologist correlation recommended."
    if region == "spine" and scores.get("disc_space_narrowing_proxy", 0.0) > 0.6:
        return "Spine radiograph suggests degenerative change proxy elevation. Clinical correlation recommended."
    return f"{region.title()} radiograph analysis complete. No high-risk pattern strongly flagged."


def _run(filepath: str, region: str) -> dict[str, Any]:
    g = _load_gray(filepath)
    scores = _bone_scores_from_gray(g, region)
    return {
        "modality": "xray",
        "detected_region": region,
        "findings": _findings(scores, region),
        "impression": _impression(scores, region),
        "pathology_scores": scores,
        "structures": {
            "narrative_report": "",
            "analysis_type": "deterministic_msk_features",
        },
        "confidence": "medium",
        "models_used": ["Manthana-MSK-Deterministic-v1"],
        "disclaimer": DISCLAIMER,
    }


def run_bone_pipeline(filepath: str, job_id: str, region: str = "extremity") -> dict:
    logger.info("[%s] Bone pipeline (region=%s) file=%s", job_id, region, filepath)
    return _run(filepath, region)


def run_spine_pipeline(filepath: str, job_id: str, region: str = "spine") -> dict:
    logger.info("[%s] Spine pipeline (region=%s) file=%s", job_id, region, filepath)
    return _run(filepath, region)


def run_skull_pipeline(filepath: str, job_id: str) -> dict:
    logger.info("[%s] Skull pipeline file=%s", job_id, filepath)
    return _run(filepath, "skull")
