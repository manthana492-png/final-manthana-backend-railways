"""
Manthana — Abdomen/Pelvis plain-film pipeline.

Deterministic image-derived proxy scoring with normalized response schema.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from disclaimer import DISCLAIMER

logger = logging.getLogger("manthana.xray.abdomen")


def _load_gray(filepath: str) -> np.ndarray:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read abdominal X-ray: {filepath}")
    return img.astype(np.float32) / 255.0


def _abdomen_scores(g: np.ndarray, region: str) -> dict[str, float]:
    h, w = g.shape
    center = g[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    lower = g[h // 2 :, :]
    edges = cv2.Canny((g * 255).astype(np.uint8), 60, 140)

    bowel_gas_proxy = float(np.clip(np.mean(center < 0.23) * 2.0, 0.0, 1.0))
    obstruction_proxy = float(np.clip(np.std(lower) * 2.2 + np.mean(edges > 0) * 0.8, 0.0, 1.0))
    calcification_proxy = float(np.clip(np.mean(g > 0.88) * 5.0, 0.0, 1.0))
    free_air_proxy = float(np.clip(np.mean(g[: h // 5, :] < 0.2) * 3.0, 0.0, 1.0))

    out = {
        "bowel_gas_pattern_proxy": bowel_gas_proxy,
        "obstruction_pattern_proxy": obstruction_proxy,
        "calcification_proxy": calcification_proxy,
        "free_air_proxy": free_air_proxy,
        "edge_density": float(np.mean(edges > 0)),
        "intensity_variance": float(np.std(g)),
    }
    if region == "pelvis":
        out["pelvic_calcification_proxy"] = float(np.clip(calcification_proxy * 1.1, 0.0, 1.0))
    return out


def _findings(scores: dict[str, float], region: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if scores.get("obstruction_pattern_proxy", 0.0) > 0.65:
        out.append(
            {
                "label": "Possible bowel obstruction pattern",
                "severity": "warning",
                "confidence": round(scores["obstruction_pattern_proxy"] * 100.0, 1),
                "region": region.title(),
                "description": "Gas-distribution and edge pattern proxy suggests possible bowel obstruction.",
            }
        )
    if scores.get("free_air_proxy", 0.0) > 0.55:
        out.append(
            {
                "label": "Possible free intraperitoneal air pattern",
                "severity": "critical",
                "confidence": round(scores["free_air_proxy"] * 100.0, 1),
                "region": region.title(),
                "description": "Upper abdominal dark-band proxy elevated; urgent radiology correlation needed.",
            }
        )
    if scores.get("calcification_proxy", 0.0) > 0.5:
        out.append(
            {
                "label": "Calcific density pattern",
                "severity": "info",
                "confidence": round(scores["calcification_proxy"] * 100.0, 1),
                "region": region.title(),
                "description": "Bright density proxy suggests possible calcific foci.",
            }
        )
    if not out:
        out.append(
            {
                "label": "No high-risk abdominal pattern",
                "severity": "clear",
                "confidence": 86.0,
                "region": region.title(),
                "description": "No major abnormality strongly flagged by deterministic feature analysis.",
            }
        )
    return out


def _impression(scores: dict[str, float], region: str) -> str:
    if scores.get("free_air_proxy", 0.0) > 0.6:
        return "Possible free-air pattern flagged on abdominal radiograph; urgent clinical/radiology correlation required."
    if scores.get("obstruction_pattern_proxy", 0.0) > 0.65:
        return "Abdominal radiograph suggests possible bowel obstruction pattern. Correlate clinically and with formal read."
    return f"{region.title()} radiograph analysis complete. No high-risk pattern strongly flagged."


def run_abdomen_pipeline(filepath: str, job_id: str, region: str = "abdomen") -> dict:
    logger.info("[%s] Abdomen pipeline (region=%s) file=%s", job_id, region, filepath)
    g = _load_gray(filepath)
    scores = _abdomen_scores(g, region)
    return {
        "modality": "xray",
        "detected_region": region,
        "findings": _findings(scores, region),
        "impression": _impression(scores, region),
        "pathology_scores": scores,
        "structures": {
            "narrative_report": "",
            "analysis_type": "deterministic_abdomen_features",
        },
        "confidence": "medium",
        "models_used": ["Manthana-Abdomen-Deterministic-v1"],
        "disclaimer": DISCLAIMER,
    }
