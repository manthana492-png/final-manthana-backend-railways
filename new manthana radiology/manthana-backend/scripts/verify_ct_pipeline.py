#!/usr/bin/env python3
"""
CT pipeline verification (Phase 12).
Run from repo: PYTHONPATH=shared python3 scripts/verify_ct_pipeline.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "shared"))


def main() -> int:
    passed = 0
    failed = 0

    def ok(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"FAIL: {msg}", file=sys.stderr)

    from schemas import AnalysisResponse, Finding

    f = Finding(
        label="Test",
        description="d",
        severity="info",
        confidence=80.0,
        region="Abdomen",
    )
    for modality in ("cardiac_ct", "spine_neuro", "abdominal_ct"):
        r = AnalysisResponse(
            job_id="test-job",
            modality=modality,
            findings=[f],
            impression="ok",
            pathology_scores={},
            confidence="medium",
            models_used=[],
        )
        ok(r.modality == modality, f"modality {modality}")
    print("AnalysisResponse schema: OK (cardiac_ct, spine_neuro, abdominal_ct)")

    from totalseg_label_maps import map_organ_key

    ok(map_organ_key("liver", "total") == "liver_cm3", "liver_cm3 map")
    ok(map_organ_key("vertebrae_L1", "vertebrae_body") == "vertebrae_L1_cm3", "L1 map")
    print("totalseg_label_maps: OK")

    from comp2comp_runner import merge_c2c_parsed_metrics

    spine = {"source": "comp2comp_spine", "Predicted T-score": 0.5, "mean_hu_trabecular": 120.0}
    lsp = {"source": "comp2comp_liver_spleen_pancreas", "liver_cm3": 1500.0}
    mat = {"source": "comp2comp_muscle_adipose", "skeletal_muscle_area_cm2": 100.0, "visceral_fat_cm2": 200.0}
    m = merge_c2c_parsed_metrics(spine, lsp, mat)
    ok(m.get("t_score_estimate") == 0.5, "merge t-score from observed header")
    ok(m.get("bmd_source") == "comp2comp_spine", "bmd_source spine")
    ok(m.get("c2c_liver_cm3") == 1500.0, "lsp liver")
    ok(m.get("muscle_area_cm2") == 100.0, "mat muscle")
    print("comp2comp_runner.merge_c2c_parsed_metrics: OK (sample headers)")

    print(f"{passed} passed" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
