#!/usr/bin/env python3
"""
Production smoke test for the CXR pipeline (TorchXRayVision ensemble + Kimi narrative policy).

Usage:
  cd manthana-backend
  PYTHONPATH=shared:services/01_body_xray python3 scripts/smoke_test_cxr.py image1.jpg [image2.jpg ...]

Requires:
  - PYTHONPATH including shared/ and services/01_body_xray/
  - KIMI_API_KEY (or MOONSHOT_API_KEY) configured for narrative generation
  - GPU optional (CPU works; slower)

Loads api-keys.env from this_studio if present (same search as inference: walk parents, then known workspace path).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "shared"))
sys.path.insert(0, str(_ROOT / "services" / "01_body_xray"))


def _load_api_keys_env() -> None:
    candidates: list[Path] = []
    here = Path(__file__).resolve()
    for p in here.parents:
        candidates.append(p / "api-keys.env")
    candidates.append(Path("/teamspace/studios/this_studio/api-keys.env"))
    for env_file in candidates:
        if not env_file.is_file():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k:
                    os.environ.setdefault(k, v)
        except OSError:
            continue
        return


_load_api_keys_env()

import inference  # noqa: E402  — after path + optional env
from schemas import AnalysisResponse  # noqa: E402

VALID_SEVERITIES = frozenset({"critical", "warning", "info", "clear"})


def _scores_are_numeric(d: dict) -> bool:
    for v in d.values():
        if v is None:
            continue
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def main() -> int:
    images = sys.argv[1:]
    if not images:
        print(
            "Usage: python scripts/smoke_test_cxr.py image1.jpg [image2.jpg ...]",
            file=sys.stderr,
        )
        return 2

    all_pass = True
    for img_path in images:
        p = Path(img_path)
        print(f"\n{'=' * 60}")
        print(f"IMAGE: {p}")
        print("=" * 60)

        if not p.is_file():
            print(f"❌ FAIL: file not found: {p}")
            all_pass = False
            continue

        t0 = time.perf_counter()
        job_id = str(uuid.uuid4())
        try:
            result = inference.run_pipeline(
                filepath=str(p.resolve()),
                job_id=job_id,
                patient_context={
                    "age": 45,
                    "sex": "M",
                    "clinical_history": "3-week cough",
                },
            )
            elapsed = time.perf_counter() - t0

            assert isinstance(result.get("findings"), list), "findings not list"
            assert isinstance(result.get("pathology_scores"), dict), "scores not dict"
            assert isinstance(result.get("structures"), dict), "structures not dict"
            assert _scores_are_numeric(result["pathology_scores"]), "non-numeric score"

            bad = [
                f
                for f in result["findings"]
                if f.get("severity") not in VALID_SEVERITIES
            ]
            assert not bad, f"invalid severity: {bad}"

            AnalysisResponse(**result)

            narrative = (
                result["structures"].get("narrative_report", "")
                if isinstance(result["structures"], dict)
                else ""
            )
            print(f"✅ PASS ({elapsed:.1f}s)")
            print(f"   job_id: {result.get('job_id', job_id)}")
            print(f"   models: {result.get('models_used')}")
            print(f"   findings: {len(result['findings'])}")
            for f in result["findings"][:3]:
                print(
                    f"   [{f.get('severity', ''):8}] {f.get('label')} "
                    f"({f.get('confidence', 0)}%)"
                )
            print(f"   narrative: {len(narrative)} chars")
            if narrative:
                first = narrative.splitlines()[0] if narrative else ""
                print(f"   first line: {first[:100]}")
            ps = result["pathology_scores"] or {}
            if ps:
                top_label = max(ps, key=lambda k: float(ps[k] or 0))
                top_score = float(ps[top_label] or 0)
                print(f"   top score: {top_label} = {top_score:.3f}")

        except Exception as e:
            print(f"❌ FAIL ({time.perf_counter() - t0:.1f}s): {e}")
            traceback.print_exc()
            all_pass = False

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'ALL PASS ✅' if all_pass else 'FAILURES FOUND ❌'}")
    print("Use full-resolution PNG/DICOM-derived images for clinically meaningful score spreads.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
