"""SynthSeg brain segmentation via subprocess (TensorFlow isolated from PyTorch)."""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import tempfile
from typing import Any

logger = logging.getLogger("manthana.synthseg_runner")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
SYNTHSEG_SCRIPT = os.getenv(
    "SYNTHSEG_SCRIPT",
    "/opt/SynthSeg/scripts/commands/SynthSeg_predict.py",
)


def run_synthseg(
    nifti_path: str,
    job_id: str,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """
    Run SynthSeg on a NIfTI brain volume. Returns volumes (cm³), qc_score if available.

    If SynthSeg is not installed or fails, returns available=False without raising.
    """
    if not os.path.isfile(nifti_path):
        return {"available": False, "reason": "file_missing"}
    if not os.path.isfile(SYNTHSEG_SCRIPT):
        logger.info("SynthSeg script not found at %s — skipping", SYNTHSEG_SCRIPT)
        return {"available": False, "reason": "synthseg_not_installed"}

    with tempfile.TemporaryDirectory(prefix=f"synthseg_{job_id}_") as tmp:
        seg_out = os.path.join(tmp, "seg.nii.gz")
        vol_csv = os.path.join(tmp, "volumes.csv")
        qc_csv = os.path.join(tmp, "qc.csv")
        cmd = [
            "python",
            SYNTHSEG_SCRIPT,
            "--i",
            nifti_path,
            "--o",
            seg_out,
            "--vol",
            vol_csv,
            "--qc",
            qc_csv,
            "--robust",
        ]
        if os.getenv("SYNTHSEG_FAST", "1") == "1":
            cmd.append("--fast")
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=os.path.dirname(SYNTHSEG_SCRIPT) or ".",
            )
        except subprocess.TimeoutExpired:
            return {"available": False, "reason": "timeout"}
        except Exception as e:
            logger.warning("SynthSeg subprocess error: %s", e)
            return {"available": False, "reason": "subprocess_error", "error": str(e)}

        if r.returncode != 0:
            err = (r.stderr or "")[-800:]
            logger.warning("SynthSeg failed rc=%s: %s", r.returncode, err)
            return {"available": False, "reason": "synthseg_failed", "stderr": err}

        volumes: dict[str, float] = {}
        if os.path.isfile(vol_csv):
            with open(vol_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k, v in row.items():
                        if not k or k.lower() == "subject":
                            continue
                        try:
                            val = float(v)
                            key = k if str(k).endswith("_cm3") else f"{k}_cm3"
                            volumes[key] = round(val / 1000.0, 4)
                        except (TypeError, ValueError):
                            continue

        qc_score: float | None = None
        if os.path.isfile(qc_csv):
            with open(qc_csv, newline="") as f:
                qreader = csv.DictReader(f)
                for row in qreader:
                    for cand in ("qc_score", "QC_score", "score"):
                        if cand in row and row[cand] not in (None, ""):
                            try:
                                qc_score = float(row[cand])
                            except (TypeError, ValueError):
                                pass
                            break
                    break

        return {
            "available": True,
            "volumes": volumes,
            "qc_score": qc_score,
            "model": "SynthSeg",
        }
