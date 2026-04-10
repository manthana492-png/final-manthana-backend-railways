"""
Digitization adapter: optional vendored PhysioNet ECG-Digitiser vs in-repo OpenCV digitizer.

Set ECG_DIGITISER_REPO_ROOT to a checkout of github.com/felixkrones/ECG-Digitiser when integrated.
Until then, always uses services/13_ecg/digitizer.py.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Tuple

import numpy as np

logger = logging.getLogger("manthana.ecg_digitiser_adapter")


def digitize_ecg_image_adapted(filepath: str, target_rate: int = 500) -> Tuple[np.ndarray, int]:
    repo = os.getenv("ECG_DIGITISER_REPO_ROOT", "").strip()
    script = os.getenv("ECG_DIGITISER_SCRIPT", "").strip()

    if repo and os.path.isdir(repo):
        entry = script or os.path.join(repo, "digitise.py")
        if os.path.isfile(entry):
            try:
                return _digitize_via_subprocess(entry, filepath, target_rate, cwd=repo)
            except Exception as e:
                logger.warning("External ECG-Digitiser failed (%s), falling back to OpenCV: %s", entry, e)

    from digitizer import digitize_ecg_image

    sig, fs = digitize_ecg_image(filepath, target_rate=target_rate)
    return sig, fs


def _digitize_via_subprocess(
    script_path: str,
    filepath: str,
    target_rate: int,
    cwd: str,
) -> Tuple[np.ndarray, int]:
    """
    Call upstream CLI if it supports --input/--output; expect CSV or NPZ output.
    This is a placeholder contract — adjust flags to match the vendored repo.
    """
    import tempfile

    out_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    out_csv.close()
    out_path = out_csv.name
    try:
        cmd = [
            sys.executable,
            script_path,
            "--input",
            filepath,
            "--output",
            out_path,
        ]
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            timeout=float(os.getenv("ECG_DIGITISE_TIMEOUT_SEC", "120")),
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        raise RuntimeError(f"ECG-Digitiser subprocess failed: {e}") from e

    try:
        from preprocessing.ecg_utils import read_ecg_csv

        sig, fs = read_ecg_csv(out_path)
        if target_rate and int(fs) != int(target_rate):
            from preprocessing.ecg_utils import normalize_ecg

            sig = normalize_ecg(sig, target_rate=target_rate, current_rate=float(fs))
            fs = target_rate
        return sig, int(fs)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
