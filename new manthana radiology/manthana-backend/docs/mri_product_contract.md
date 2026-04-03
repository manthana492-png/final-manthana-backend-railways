# MRI product contract (production)

Single source of truth for **what each MRI-related product does**, how it is **routed**, and what is **measured vs narrative-only**. Companion to `docs/mri_release_runbook.md`.

## 1. Product matrix

| Product (UI / gateway intent) | Gateway `modality` (examples) | Canonical service | Response `modality` | Notes |
|------------------------------|------------------------------|-------------------|---------------------|--------|
| Brain / head MRI | `brain_mri`, `mri`, `brain`, `head_mri` | `02_brain_mri` | `brain_mri` | Aliases `mri` / `brain` / `head_mri` resolve to `brain_mri`. |
| Spine / neuro MRI | `spine_neuro`, `spine_mri`, `spine`, `neuro` | `10_spine_neuro` | `spine_neuro` | Use **`spine_mri`** for explicit MRI spine routing; TotalSeg task `vertebrae_mr` when MR is detected. |
| MSK MRI (knee, shoulder, etc.) | PACS-routed `unsupported_mr_msk` | Gateway stub | N/A | **Not supported** in v1 — deterministic unsupported response. |

**Important:** The alias **`mri` always routes to the brain MRI service** (`brain_mri`). It does **not** auto-route spine or MSK studies. Users must pick **Spine / Neuro MRI** for spine imaging.

## 2. Brain MRI pipeline (`02_brain_mri`)

| Component | Role | Availability |
|-----------|------|--------------|
| TotalSegmentator `total_mr` | Multi-structure MR segmentation + organ/structure volumes (`*_cm3` keys) | Required for `/ready` when `totalsegmentator` import succeeds |
| SynthSeg | Brain sub-structure parcellation volumes | Optional — requires NIfTI + `SYNTHSEG_SCRIPT` |
| Prima | Study-level diagnostic scores | Optional — requires `PRIMA_CONFIG_YAML` + mounted weights |
| Narrative | Kimi → Anthropic text (optional vision: middle axial slice PNG) | Policy: `MRI_NARRATIVE_POLICY` |

### Input quality

| Input | Supported | Notes |
|-------|-----------|-------|
| NIfTI (`.nii`, `.nii.gz`) | Yes | Preferred for SynthSeg + TotalSeg |
| DICOM series (`series_dir` + first instance upload) | Yes | Converted to NIfTI for TotalSeg/SynthSeg |
| Single-slice / 2D raster (PNG/JPG) | Degraded | TotalSeg/SynthSeg volumetric paths skipped or limited; findings state limitations |

### Metric provenance

| Output | Type |
|--------|------|
| `pathology_scores` numeric volumes | **Measurement** from segmentation masks when segmentation succeeds |
| SynthSeg QC / volumes | **Model output** — correlate with visual read |
| Prima logits | **Model output** — optional; not a standalone diagnosis |
| `structures.narrative_report` | **Not a measurement** — LLM synthesis from JSON + optional slice image |

## 3. Narrative policy (env-controlled)

| Variable | Service | Default | Values |
|----------|---------|---------|--------|
| `MRI_NARRATIVE_POLICY` | `02_brain_mri` | `kimi_then_anthropic` | `kimi_then_anthropic`, `kimi_only`, `anthropic_only`, `off` |
| `KIMI_MRI_MODEL` | `02_brain_mri` | `moonshot-v1-8k` | Moonshot OpenAI-compatible model id |
| `KIMI_BASE_URL` | `02_brain_mri` | `https://api.moonshot.ai/v1` | Override for compatible endpoints |
| `MRI_NARRATIVE_VISION` | `02_brain_mri` | `1` | `0` = text-only to Kimi (no middle-slice image upload to LLM) |

Failures are **non-fatal** unless operations add stricter CI checks.

## 4. PHI / third-party LLM

When narrative is enabled, the service may send to Kimi/Anthropic:

- Structured `pathology_scores` and findings JSON
- Pipeline `impression`
- Parsed `patient_context` from `clinical_notes` / `patient_context_json`
- Optional: one grayscale PNG (middle axial slice) when `MRI_NARRATIVE_VISION=1`

Operators must have appropriate **agreements and policies** before processing identifiable PHI.

## 5. Gateway transparency fields (MRI deep responses)

Populated by the gateway for `brain_mri` responses:

- `mri_product` — `brain_mri`
- `gateway_request_modality` — raw `modality` form field
- `mri_routing_note` — short routing explanation (e.g. alias resolution)

Spine/neuro MRI via `spine_neuro` uses existing **CT-style** transparency fields (`ct_product` is not set for pure MRI spine path; `modality` remains `spine_neuro`).

## 6. GPU / TotalSeg idle

Brain MRI uses `shared/totalseg_runner.py`, which honors:

- `CT_TOTALSEG_IDLE_EMPTY_CACHE_SEC`
- `CT_TOTALSEG_IDLE_CHECK_SEC`

These names are **shared with CT**; behavior is identical (optional `torch.cuda.empty_cache()` after idle).

## 7. Definition of done (this document)

- [x] No claim that generic `mri` auto-detects spine vs brain.
- [x] Spine MRI has an explicit gateway path (`spine_mri` → `spine_neuro`).
- [x] MSK MRI unsupported path is documented.
- [x] Narrative and vision are policy-controlled and documented.
