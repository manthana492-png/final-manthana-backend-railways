#!/usr/bin/env python3
"""Single DICOM-series inference (must run as file — TotalSeg/nnUNet multiprocessing)."""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "services", "02_brain_mri"))

os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("TOTALSEG_DEVICE", "cpu")

from schemas import AnalysisResponse  # noqa: E402

import inference  # noqa: E402


def main() -> int:
    p = "/tmp/manthana_e2e_mri/mri_dicom_series"
    t0 = time.time()
    r = inference.run_pipeline(
        filepath=p,
        job_id="e2e_dicom_series",
        patient_context={"clinical_history": "headache"},
    )
    AnalysisResponse(**r)
    print(f"PASS DICOM series {time.time() - t0:.1f}s models={r.get('models_used')} conf={r.get('confidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
