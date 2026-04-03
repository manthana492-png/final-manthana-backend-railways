#!/usr/bin/env python3
"""E2E brain MRI inference (run as file — avoids TotalSeg/nnUNet spawn issues with stdin)."""

from __future__ import annotations

import glob
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "services", "02_brain_mri"))

from schemas import AnalysisResponse  # noqa: E402

import inference  # noqa: E402


def main() -> int:
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("TOTALSEG_DEVICE", "cpu")

    env_file = "/teamspace/studios/this_studio/api-keys.env"
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

    test_files = (
        sorted(glob.glob("/tmp/manthana_e2e_mri/*.nii*"))
        + sorted(glob.glob("/tmp/manthana_e2e_mri/*.png"))
        + ["/tmp/manthana_e2e_mri/mri_dicom_series"]
    )
    test_files = [f for f in test_files if os.path.exists(f)][:3]

    print(f"Testing {len(test_files)} MRI inputs:")
    for f in test_files:
        print(f"  {f}")

    VALID_SEV = {"critical", "warning", "info", "clear"}
    all_pass = True

    for i, mri_path in enumerate(test_files, 1):
        print(f"\n{'='*60}")
        print(f"MRI INPUT {i}: {os.path.basename(mri_path)}")
        print(f"Type: {'DICOM series directory' if os.path.isdir(mri_path) else 'file'}")
        print("=" * 60)
        t0 = time.time()

        try:
            result = inference.run_pipeline(
                filepath=mri_path,
                job_id=f"mri_test_{i}",
                patient_context={
                    "age": 35,
                    "sex": "F",
                    "clinical_history": "headache, seizure, fever",
                    "indication": "rule out intracranial pathology",
                    "mri_sequence": "T1",
                },
            )
            elapsed = time.time() - t0

            assert isinstance(result["findings"], list)
            assert isinstance(result["pathology_scores"], dict)
            assert isinstance(result["structures"], dict)

            bad_scores = [
                (k, type(v))
                for k, v in result["pathology_scores"].items()
                if not isinstance(v, float)
            ]
            assert not bad_scores, f"non-float scores: {bad_scores}"

            bad_sev = [
                f.get("label")
                for f in result["findings"]
                if f.get("severity") not in VALID_SEV
            ]
            assert not bad_sev, f"bad severity: {bad_sev}"

            AnalysisResponse(**result)

            narrative = result["structures"].get("narrative_report", "")
            emergency = result["structures"].get("emergency_flags", [])

            print(f"PASS ({elapsed:.1f}s)")
            print(f"   confidence: {result.get('confidence')}")
            print(f"   models: {result.get('models_used', [])}")
            print(f"   findings ({len(result['findings'])}):")
            for f in result["findings"][:6]:
                print(
                    f"     [{f.get('severity', '?'):8}] {f.get('label', '?')} "
                    f"({f.get('confidence', 0)}%)"
                )
            print(f"   pathology_scores (sample): {list(result['pathology_scores'].items())[:6]}")
            print(f"   structures keys: {list(result['structures'].keys())}")
            if emergency:
                print(f"   EMERGENCY FLAGS: {emergency}")
            print(f"   narrative: {len(narrative)} chars")
            if narrative:
                print(f"   preview: {narrative[:220]}...")

        except Exception as e:
            print(f"FAIL ({time.time() - t0:.1f}s): {e}")
            import traceback

            traceback.print_exc()
            all_pass = False

    print(f"\n{'='*60}")
    print(f"RESULT: {'ALL PASS' if all_pass else 'FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
