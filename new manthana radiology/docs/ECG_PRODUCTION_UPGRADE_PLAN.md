# ECG Production Implementation Plan (Hardened)

This document mirrors the Cursor plan **ECG Production Upgrade** with operational caveats resolved: **Modal CPU–wise defaults**, **hybrid fallback**, **versioned label mapping**, **real integration paths** (no fictional PyPI packages), and **phased rollout**.  
**Source of truth for Cursor todos:** `.cursor/plans/ecg_production_upgrade_f25f9582.plan.md` (if present on your machine).

## Executive Summary

**Current State:** Predictable CPU pipeline — OpenCV-style digitization, nine heuristic rhythm scores, neurokit2 intervals, optional OpenRouter narrative (`services/13_ecg/inference.py`).

**Target State:** **Optional** deep-learning stages — ECG-Digitiser (PhysioNet 2024 family, BSD) for robust **image → 12-lead signal**, ECGFounder (Apache-2.0 family) for **many-label** scores — while **default production behavior remains hybrid**: DL when weights and SLO allow, **automatic fallback** to today’s heuristics.

**Why this revision:** Closes gaps around **dependencies**, **weight hosting**, **latency/memory on Modal CPU**, **phone-photo validation**, **correct label mapping into the existing API**, and **LazyModel vs custom wrappers**.

---

## Caveats Addressed

| Caveat | Resolution |
|--------|------------|
| Heavy dependencies | Layered image; optional `requirements-ecg-dl.txt`; `ECG_DL_TIER=off` skips DL imports. |
| Weight hosting | Volume paths `/models/ecg/digitiser/`, `/models/ecg/founder/` + `manifest.json` (version, SHA256); **lazy load** on first inference. |
| Latency / memory (Modal CPU) | Profiled `cpu`/`memory`; thread caps (`OMP_NUM_THREADS`, `torch.set_num_threads`); per-stage timeouts → fallback; `structures.ecg_timing_ms`; optional ONNX if upstream ships it. |
| Phone photo validation | Pre-digitization **quality gate** (blur, resolution, exposure); golden set in CI; staging before prod. |
| Label mapping / API | Versioned `ecg_label_map.v1.json`; `structures.ecg_pipeline_version`; tests for legacy `RHYTHM_KEYS`; `models_used` lists actual branch. |
| LazyModel may not fit | Use `LazyModel` only for HF-compatible checkpoints; else `EcgFounderWrapper` + explicit `torch.load` / `forward` from official code. |
| Predictability at scale | Default **`ECG_MODE=hybrid`**; **`legacy`** forces current behavior; **`deep_learning`** only when hard-fail without weights is intentional. |

---

## Integration Reality (No Fictional Packages)

Do **not** assume `pip install ecg_digitiser` or `from ecgfounder import ECGFounder` are stable PyPI APIs.

1. **ECG-Digitiser:** Pin a **commit SHA** of `github.com/felixkrones/ECG-Digitiser`. Integrate via **git submodule** under `third_party/ECG-Digitiser` or invoke documented CLI/`digitise.py`. Wrap in an adapter exposing the same `(filepath) -> (np.ndarray 12×T, fs)` as `digitizer.py`.

2. **ECGFounder:** Use **official** repo / HuggingFace if available; else **clone + state_dict**. Implement `shared/ecg_founder_infer.py` with `predict_12lead(signal, fs) -> dict[str, float]`.

3. **CI:** Mock adapters so tests do not require multi-GB weights.

---

## Modal CPU: Wise Defaults

Start **cpu=2.0**, **memory=4096**; raise only after **OOM** or **p95 SLO** failure in staging. Use **torch CPU** wheels only. Cap threads to avoid oversubscription on 2 vCPU. Log **`ecg_timing_ms`** per stage.

---

## Phased Rollout

| Phase | Scope | “Production-ready” when |
|-------|--------|-------------------------|
| 0 | Current heuristic + OpenCV | Already |
| 1 | ECGFounder on **digital** inputs only | Weights on volume; CPU SLO met; mapping v1 tested |
| 2 | ECG-Digitiser + founder on **photos** | Image gate + golden photos; fallback verified |

---

## Configuration (Conceptual)

- `ECG_MODE`: `hybrid` | `legacy` | `deep_learning`
- `ECG_DL_TIER`: `off` | `founder_only` | `full`
- Timeouts, paths under `/models/ecg/`, image gate thresholds — see future `services/13_ecg/config.py`.

---

## Deployment Pattern (Unchanged)

**Vercel** (UI) → **Railway** gateway (`ECG_SERVICE_URL`) → **Modal CPU** ECG app with **volume `/models`**, scale-to-zero — same as today; image size/memory may increase with DL extras.

---

## Files to Implement

See the Cursor plan file for the full **files to modify** list and **success criteria**.

---

## Honest Scope

**Production-ready** = staged rollout + hybrid fallback + observability + versioned mapping — not a single-day swap of heuristics for DL everywhere. CDS positioning unchanged unless you pursue regulatory clearance separately.
