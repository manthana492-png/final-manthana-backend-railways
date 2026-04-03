# CT product contract (production)

Single source of truth for **what each CT sub-product does**, how it is **routed**, and **what is measured vs heuristic**. Used by backend, frontend, QA, and release gates.

## 1. Product matrix

| Product (UI / gateway intent) | Gateway `modality` (examples) | Canonical service | Response `modality` | Chest-thoracic alias |
|------------------------------|------------------------------|-------------------|----------------------|----------------------|
| Abdomen / pelvis CT | `abdominal_ct`, `ct`, `ct_scan`, `abdomen` | `08_abdominal_ct` | `abdominal_ct` | No |
| Thoracic (chest) CT | `chest_ct` | `08_abdominal_ct` | `abdominal_ct` | **Yes** — same service; use `patient_context_json.ct_region=chest_ct` |
| Cardiac CT | `cardiac_ct`, `heart`, `cardiac` | `04_cardiac_ct` | `cardiac_ct` | No |
| Spine / neuro CT | `spine_ct`, `spine`, `neuro` | `10_spine_neuro` | `spine_neuro` | No |
| **Brain / head NCCT** | `ct_brain`, `brain_ct`, `head_ct`, `ncct_brain` | `11_ct_brain` | `ct_brain` | No |

**Important:** Thoracic CT is **not** a separate Docker service. It is **intentionally** routed to the abdominal CT pipeline with `ct_region` in context so outputs and narrative stay consistent with the shared TotalSegmentator + Comp2Comp stack.

## 2. Triage

- **X-ray** may short-circuit on triage. **CT does not** — non–X-ray triage is pass-through to deep analysis.

## 3. Input quality bands (abdominal / chest alias path)

| Band (declared slices / path) | TotalSegmentator | Organ volumes | Comp2Comp | Narrative |
|------------------------------|------------------|---------------|-----------|-----------|
| Raster (JPG/PNG) | Visual / degraded path | Not true volumetrics | Optional / limited | If policy allows |
| DICOM &lt; ~30 slices | Often skipped or fast | Unreliable | May skip | If policy allows |
| ~30–80 slices | `fast` mode typical | Approximate | Partial | If policy allows |
| 80+ slices | `full` typical | Stronger | Full-series when available | If policy allows |

Exact thresholds are driven by `patient_context_json` (`declared_file_count`, `totalseg_model`, `upload_type`) and runtime series detection in `08_abdominal_ct`.

### 3b. CT Brain (NCCT) — Phase 1

| Input | Supported | Notes |
|-------|-----------|--------|
| DICOM series directory (`series_dir`) or single DICOM / NIfTI | Yes | Prefer full axial NCCT head series |
| Raster screenshots (JPG/PNG) | Degraded | Single-slice / 2D only; no volumetric ICH claims |
| CTA / CTP / contrast brain CT | Out of scope v1 | Route to future products; service may warn if contrast implied |

**Inference modes (deploy-only, no training in-repo):**

- `torchscript` — `CT_BRAIN_TORCHSCRIPT_PATH` points to a **clinically validated** TorchScript bundle you mount at deploy time.
- `ci_dummy` — `CT_BRAIN_CI_DUMMY_MODEL=1` for automated tests only; **not** for patient use.
- `weights_required` — default when no TorchScript path: structured response states model weights are not configured; **no** hemorrhage probability is reported.

**Diagnostic-claim governance:** performance thresholds, external validation, and human-in-the-loop review for positive/low-confidence cases are **mandatory** before marketing as diagnostic (see `docs/ct_release_runbook.md`).

## 4. Metric provenance

| Metric | Abdominal / chest alias | Cardiac | Spine / neuro |
|--------|-------------------------|---------|---------------|
| Organ / structure volumes from masks | **Measurement** (voxel × spacing) when masks exist | **Measurement** for segmented structures | **Measurement** where masks exist |
| Aortic diameter (mask-based) | **Proxy** — algorithmic from aorta mask; oblique acquisition caveat | **Proxy** — same | N/A |
| Comp2Comp FDA-ref metrics | **Regulatory-labelled pipeline output** when Comp2Comp runs; availability not guaranteed | N/A | Spine-density variants when enabled |
| LLM narrative | **Not a measurement** — synthesis from structured JSON only | Same | Same |
| CT Brain `ich_probability` / subtype scores | **Classifier output** (calibrated only if your deployed weights are calibrated) | N/A | N/A |
| CT Brain heuristic / CI dummy | **Not for clinical diagnostic use** — tests or explicit staging flag only | N/A | N/A |

## 5. Narrative policy (env-controlled)

| Subtype | Env | Default | Meaning |
|---------|-----|---------|---------|
| Abdominal (+ chest alias) | `CT_ABDOMINAL_NARRATIVE_POLICY` | `kimi_then_anthropic` | `kimi_only` = Kimi only; `off` = no narrative |
| Cardiac | `CT_CARDIAC_NARRATIVE_POLICY` | `off` | `kimi_then_anthropic` enables Kimi → Anthropic when keys present |
| Spine / neuro | `CT_SPINE_NARRATIVE_POLICY` | `kimi_then_anthropic` | `kimi_only`, `anthropic_only`, `off` |
| CT Brain (NCCT) | `CT_BRAIN_NARRATIVE_POLICY` | `kimi_then_anthropic` | `kimi_only`, `off` |

Failures must be **non-fatal**; empty narrative is allowed unless ops explicitly configures stricter checks in CI.

**CT Brain critical policy:** If `ich_probability` (or equivalent) from the **deployed TorchScript model** exceeds `CT_BRAIN_CRITICAL_THRESHOLD` (default `0.5`), a **critical** structured finding is emitted **even when narrative generation fails**.

## 6. Gateway transparency fields (deep CT responses)

Populated by the gateway for CT family responses:

- `ct_product` — canonical service modality: `abdominal_ct` | `cardiac_ct` | `spine_neuro` | `ct_brain`
- `gateway_request_modality` — raw `modality` form field from the client
- `ct_region_context` — `patient_context_json.ct_region` when present
- `ct_routing_note` — short human-readable routing explanation (e.g. chest alias)

## 7. GPU idle behaviour (TotalSegmentator family)

TotalSegmentator is invoked via upstream libraries; **full weight eviction** like custom TXRV singletons is not guaranteed. When `CT_TOTALSEG_IDLE_EMPTY_CACHE_SEC > 0`, services call `torch.cuda.empty_cache()` after idle intervals to reduce VRAM pressure. See `docs/ct_release_runbook.md` for SLO-style expectations.

## 7b. CT Brain GPU idle

CT Brain uses `CT_BRAIN_GPU_IDLE_EMPTY_CACHE_SEC` / `CT_BRAIN_GPU_IDLE_CHECK_SEC` (same pattern as TotalSeg idle: optional `torch.cuda.empty_cache()` after idle; does not guarantee full weight unload).

## 8. Definition of done (this document)

- [x] No ambiguous “one CT does everything with identical guarantees” language without the caveats above.
- [x] Chest CT alias to abdominal service is **explicit**, not hidden.
- [x] Cardiac and spine products have **separate** services and contracts.
- [x] CT Brain (NCCT) is a **separate** product with explicit inference modes and diagnostic-claim guardrails.
