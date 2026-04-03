"""Comp2Comp CLI integration (--input_path) + HU fallbacks when no series dir."""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import subprocess
import sys
from typing import Any

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger("manthana.comp2comp_runner")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
COMP2COMP_DIR = os.getenv("COMP2COMP_DIR", "/opt/Comp2Comp")
C2C_TIMEOUT_SEC = int(os.getenv("C2C_TIMEOUT_SEC", "600"))


def _c2c_executable() -> str:
    return os.path.join(COMP2COMP_DIR, "bin", "C2C")


def run_c2c_pipeline(pipeline: str, dicom_series_dir: str) -> dict[str, Any]:
    """
    Run official Comp2Comp CLI: bin/C2C <pipeline> --input_path <dir> (or -i per upstream README).
    """
    exe = _c2c_executable()
    if not os.path.isfile(exe):
        logger.warning("Comp2Comp binary not found at %s", exe)
        return {"source": "c2c_unavailable", "error": "binary_missing"}

    in_path = os.path.abspath(dicom_series_dir)
    last_err = ""
    # Prefer --input_path; retry with -i (documented in Comp2Comp README) if CLI rejects first form.
    for flag in ("--input_path", "-i"):
        cmd = [exe, pipeline, flag, in_path]
        try:
            r = subprocess.run(
                cmd,
                cwd=COMP2COMP_DIR,
                capture_output=True,
                text=True,
                timeout=C2C_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            logger.warning("C2C %s timed out", pipeline)
            return {"source": "c2c_timeout"}
        except Exception as e:
            logger.warning("C2C %s error: %s", pipeline, e)
            return {"source": "c2c_error", "error": str(e)}
        if r.returncode == 0:
            break
        last_err = (r.stderr or "")[:2000]
    else:
        logger.warning("C2C %s failed (tried --input_path and -i): %s", pipeline, last_err)
        return {"source": "c2c_failed", "stderr": last_err[:500]}

    return _parse_c2c_outputs(dicom_series_dir, pipeline)


def _parse_c2c_outputs(base_dir: str, pipeline: str) -> dict[str, Any]:
    """Collect metrics from CSV/JSON under base_dir (spike-verified layout may vary)."""
    out: dict[str, Any] = {"source": f"comp2comp_{pipeline}", "pipeline": pipeline}
    for ext in ("*.csv", "*.json"):
        for path in glob.glob(os.path.join(base_dir, "**", ext), recursive=True):
            try:
                if path.endswith(".csv"):
                    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            for k, v in row.items():
                                if k is None:
                                    continue
                                key = str(k).strip()
                                if not key:
                                    continue
                                try:
                                    val = float(v) if v not in (None, "") else None
                                except (TypeError, ValueError):
                                    val = v
                                out[key] = val
                                # Duplicate under normalized key so merge can match observed CSV headers.
                                nk = "".join(
                                    c.lower() if c.isalnum() else "_" for c in key
                                ).strip("_")
                                if nk and nk != key.lower():
                                    out.setdefault(nk, val)
                elif path.endswith(".json"):
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            out.update(data)
            except Exception as e:
                logger.debug("Skip parse %s: %s", path, e)
    return out


def _heuristic_aorta(volume: np.ndarray, spacing_mm: float) -> dict[str, Any]:
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim == 2:
        v = v[..., np.newaxis]
    elif v.ndim < 2:
        return {
            "max_aorta_diameter_mm": 25.0,
            "aaa_detected": False,
            "source": "heuristic_hu",
        }
    flat = v.ravel()
    p99 = float(np.percentile(flat, 99))
    thresh = max(100.0, min(300.0, p99 * 0.35))
    mask = (v > thresh).astype(np.float32)
    if mask.sum() < 10:
        max_d_mm = 25.0
    else:
        z_best = 0.0
        for z in range(mask.shape[2]):
            layer = mask[:, :, z]
            if layer.sum() < 1:
                continue
            ys, xs = np.where(layer > 0.5)
            dy = (ys.max() - ys.min() + 1) * spacing_mm
            dx = (xs.max() - xs.min() + 1) * spacing_mm
            z_best = max(z_best, max(dx, dy))
        max_d_mm = float(max(18.0, min(60.0, z_best or 28.0)))

    aaa = max_d_mm >= 30.0
    return {
        "max_aorta_diameter_mm": round(max_d_mm, 2),
        "aaa_detected": aaa,
        "source": "heuristic_hu",
    }


def _heuristic_bmd(volume: np.ndarray) -> dict[str, Any]:
    v = np.asarray(volume, dtype=np.float32)
    bone = v > 150
    if not np.any(bone):
        hu_mean = 120.0
    else:
        hu_mean = float(np.mean(v[bone]))

    score = float(np.clip((hu_mean - 50.0) / 2.5, 0.0, 100.0))
    low = score < 45.0
    t_est = float(np.clip((score - 50.0) / 15.0, -3.0, 3.0))
    return {
        "bmd_score": round(score, 2),
        "low_bmd_flag": low,
        "t_score_estimate": round(t_est, 2),
        "muscle_area_cm2": None,
        "visceral_fat_cm2": None,
        "source": "heuristic_hu",
    }


def _load_volume_any(path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    ext = path.lower().split(".")[-1]
    try:
        if ext in ("nii", "gz"):
            from preprocessing.nifti_utils import read_nifti

            vol, aff = read_nifti(path)
            return vol, aff
        from preprocessing.dicom_utils import read_dicom

        arr, _ = read_dicom(path)
        return arr, np.eye(4)
    except Exception as e:
        logger.warning("Could not load volume: %s", e)
        return None, None


def _spacing_mm_from_affine(affine: np.ndarray | None) -> float:
    if affine is None:
        return 1.0
    return float(np.linalg.norm(affine[:3, 2]))


def run_aaq(
    ct_volume_path: str | None = None,
    volume: np.ndarray | None = None,
    affine: np.ndarray | None = None,
) -> dict[str, Any]:
    if volume is None and ct_volume_path:
        volume, affine = _load_volume_any(ct_volume_path)

    if volume is None:
        return {
            "max_aorta_diameter_mm": 0.0,
            "aaa_detected": False,
            "source": "none",
        }

    spacing = _spacing_mm_from_affine(affine)
    out = _heuristic_aorta(volume, spacing)
    out["source"] = out.get("source", "heuristic_hu")
    return out


def run_bmd(
    ct_volume_path: str | None = None,
    volume: np.ndarray | None = None,
    affine: np.ndarray | None = None,
) -> dict[str, Any]:
    if volume is None and ct_volume_path:
        volume, affine = _load_volume_any(ct_volume_path)

    if volume is None:
        return {
            "bmd_score": 0.0,
            "low_bmd_flag": True,
            "t_score_estimate": 0.0,
            "source": "none",
        }

    out = _heuristic_bmd(volume)
    _ = affine
    return out


def merge_c2c_parsed_metrics(
    spine: dict[str, Any],
    lsp: dict[str, Any],
    mat: dict[str, Any],
) -> dict[str, Any]:
    """Normalize merged CSV keys into pathology-friendly fields (best-effort)."""
    merged: dict[str, Any] = {"bmd_source": "unavailable", "lsp_source": "unavailable", "mat_source": "unavailable"}

    def pick_float(d: dict, *keys: str) -> float | None:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    continue
        return None

    def pick_float_substrings(d: dict, *must_contain: str) -> float | None:
        """First numeric whose key (lowercased, non-alnum -> _) contains every substring."""
        for k, v in d.items():
            if not isinstance(k, str) or v is None or v == "":
                continue
            kl = "".join(c.lower() if c.isalnum() else "_" for c in k).strip("_")
            if all(s in kl for s in must_contain):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    # Spine BMD
    if spine.get("source", "").startswith("comp2comp"):
        merged["bmd_source"] = "comp2comp_spine"
        merged["bmd_score"] = pick_float(spine, "bmd_score", "BMD", "Mean HU", "mean_hu", "mean_hu_trabecular")
        if merged.get("bmd_score") is None:
            merged["bmd_score"] = pick_float_substrings(spine, "mean", "hu")
        merged["t_score_estimate"] = pick_float(
            spine, "T-score", "T_score", "t_score", "pred_t_score", "predicted_t_score"
        )
        if merged.get("t_score_estimate") is None:
            merged["t_score_estimate"] = pick_float_substrings(spine, "predicted", "t", "score")
        if merged.get("t_score_estimate") is None:
            merged["t_score_estimate"] = pick_float_substrings(spine, "t", "score")
        ts = merged.get("t_score_estimate")
        if ts is not None:
            merged["low_bmd_flag"] = ts < -1.0
            merged["osteoporosis_flag"] = ts < -2.5

    # Liver/spleen/pancreas volumes
    if lsp.get("source", "").startswith("comp2comp"):
        merged["lsp_source"] = "comp2comp_liver_spleen_pancreas"
        for k in ("liver", "spleen", "pancreas"):
            v = pick_float(lsp, f"{k}_cm3", f"{k}_volume_cm3", k, f"{k}_volume")
            if v is None:
                v = pick_float_substrings(lsp, k, "cm3")
            if v is not None:
                merged[f"c2c_{k}_cm3"] = v

    # Muscle / adipose
    if mat.get("source", "").startswith("comp2comp"):
        merged["mat_source"] = "comp2comp_muscle_adipose"
        m = pick_float(mat, "muscle_area_cm2", "muscle_cm2", "skeletal_muscle_area_cm2")
        if m is None:
            m = pick_float_substrings(mat, "muscle", "cm2")
        vf = pick_float(mat, "visceral_fat_cm2", "vat_cm2", "visceral_adipose_cm2")
        if vf is None:
            vf = pick_float_substrings(mat, "visceral", "fat")
        if vf is None:
            vf = pick_float_substrings(mat, "vat")
        if m is not None:
            merged["muscle_area_cm2"] = m
        if vf is not None:
            merged["visceral_fat_cm2"] = vf

    return merged


def run_comp2comp_series(
    series_dir: str | None,
    run_spine: bool = True,
    run_lsp: bool = True,
    run_mat: bool = True,
    run_contrast: bool = False,
) -> dict[str, Any]:
    """
    Run Comp2Comp pipelines on a DICOM series directory. No-op if series_dir missing.
    """
    if not series_dir or not os.path.isdir(series_dir):
        return {
            "source": "unavailable_single_file",
            "note": "Comp2Comp requires a full DICOM series directory (e.g. PACS export).",
        }

    spine = run_c2c_pipeline("spine", series_dir) if run_spine else {}
    lsp = run_c2c_pipeline("liver_spleen_pancreas", series_dir) if run_lsp else {}
    # Official pipeline name (README): spine_muscle_adipose_tissue
    mat = run_c2c_pipeline("spine_muscle_adipose_tissue", series_dir) if run_mat else {}
    contrast = run_c2c_pipeline("contrast_phase", series_dir) if run_contrast else {}

    merged = merge_c2c_parsed_metrics(spine, lsp, mat)
    merged["source"] = "comp2comp_series"
    merged["contrast_phase_raw"] = contrast if run_contrast else None
    merged["series_dir"] = series_dir
    return merged


def run_comp2comp_abdominal(
    ct_volume_path: str,
    volume: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    series_dir: str | None = None,
) -> dict[str, Any]:
    """
    Combined metrics for abdominal CT: mask-based / heuristics + optional Comp2Comp series.
    """
    vol = volume
    aff = affine
    if vol is None:
        vol, aff = _load_volume_any(ct_volume_path)

    aaq = run_aaq(ct_volume_path=ct_volume_path, volume=vol, affine=aff)
    bmd = run_bmd(ct_volume_path=ct_volume_path, volume=vol, affine=aff)

    base = {
        "max_aorta_diameter_mm": aaq.get("max_aorta_diameter_mm"),
        "aaa_detected": aaq.get("aaa_detected", False),
        "bmd_score": bmd.get("bmd_score"),
        "low_bmd_flag": bmd.get("low_bmd_flag", False),
        "t_score_estimate": bmd.get("t_score_estimate"),
        "muscle_area_cm2": bmd.get("muscle_area_cm2"),
        "visceral_fat_cm2": bmd.get("visceral_fat_cm2"),
        "aaq_source": aaq.get("source"),
        "bmd_source": bmd.get("source"),
    }

    if series_dir and os.path.isdir(series_dir):
        c2c = run_comp2comp_series(series_dir)
        base["c2c_series"] = c2c
        # Overlay Comp2Comp values when present
        for k, v in c2c.items():
            if k in (
                "bmd_score",
                "t_score_estimate",
                "low_bmd_flag",
                "muscle_area_cm2",
                "visceral_fat_cm2",
                "osteoporosis_flag",
            ) and v is not None:
                base[k] = v
        if c2c.get("bmd_source") == "comp2comp_spine":
            base["bmd_source"] = "comp2comp_spine"
        if c2c.get("mat_source") == "comp2comp_muscle_adipose":
            base["muscle_area_cm2"] = c2c.get("muscle_area_cm2") or base.get("muscle_area_cm2")
            base["visceral_fat_cm2"] = c2c.get("visceral_fat_cm2") or base.get("visceral_fat_cm2")

    return base
