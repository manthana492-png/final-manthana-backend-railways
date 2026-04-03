# CT release, monitoring, and rollback

Companion to `docs/ct_product_contract.md`. Use this for staged rollout of **abdominal/chest-alias**, **cardiac**, **spine/neuro**, and **CT Brain (NCCT)** sub-products.

## 1. Environment flags (per subtype)

| Variable | Service(s) | Purpose |
|----------|------------|---------|
| `CT_ABDOMINAL_NARRATIVE_POLICY` | `08_abdominal_ct` | `kimi_then_anthropic` (default), `kimi_only`, `off` |
| `CT_CARDIAC_NARRATIVE_POLICY` | `04_cardiac_ct` | `off` (default), `kimi_then_anthropic`, `kimi_only` |
| `CT_SPINE_NARRATIVE_POLICY` | `10_spine_neuro` | `kimi_then_anthropic` (default), `kimi_only`, `anthropic_only`, `off` |
| `CT_BRAIN_NARRATIVE_POLICY` | `11_ct_brain` | `kimi_then_anthropic` (default), `kimi_only`, `off` |
| `CT_BRAIN_TORCHSCRIPT_PATH` | `11_ct_brain` | Path to **validated** TorchScript bundle (mounted at deploy). If unset, service returns `weights_required` mode (no ICH scores). |
| `CT_BRAIN_CRITICAL_THRESHOLD` | `11_ct_brain` | Probability above which a **critical** finding is forced (default `0.5`). |
| `CT_BRAIN_CI_DUMMY_MODEL` | `11_ct_brain` | `1` = CI-only dummy forward pass; **never** in patient-facing deploy. |
| `CT_BRAIN_GPU_IDLE_EMPTY_CACHE_SEC` | `11_ct_brain` | `0` = disabled; `>0` = idle `torch.cuda.empty_cache()` |
| `CT_BRAIN_GPU_IDLE_CHECK_SEC` | `11_ct_brain` | Reaper interval (default `30`) |
| `CT_TOTALSEG_IDLE_EMPTY_CACHE_SEC` | any service using `totalseg_runner` | `0` = disabled; `>0` = after idle, `torch.cuda.empty_cache()` |
| `CT_TOTALSEG_IDLE_CHECK_SEC` | same | Background reaper sleep interval (default `30`) |

API keys (`KIMI_*`, `ANTHROPIC_*`) behave as today; missing keys produce empty narrative (non-fatal) except where policy is `off`.

## 2. Gateway transparency

Deep CT responses include:

- `ct_product`, `gateway_request_modality`, optional `ct_region_context`, `ct_subtype`, `ct_routing_note`

**Monitoring / dashboards:** group metrics by `ct_product` and `ct_subtype`, not a single `CT` bucket.

### 2b. Diagnostic-claim gates (CT Brain)

Before marketing CT Brain as **diagnostic**:

- Locked validation set metrics (sensitivity/specificity/AUC or agreed clinical endpoints) signed off by clinical governance.
- Multi-site / multi-scanner external validation.
- Human-in-the-loop review queue for: model-positive, low-confidence, and critical-flag cases.
- Model + software version pinning in every response (`structures.algorithm_version`).
- `CT_BRAIN_CI_DUMMY_MODEL` must be **absent or 0** in production.

## 3. Canary gates (suggested)

Per subtype, for 24–72h after deploy:

| Signal | Gate |
|--------|------|
| HTTP 5xx rate | &lt; 1% of CT requests for that subtype |
| p95 latency | Within SLO agreed per subtype (document baseline pre-release) |
| Empty `findings` with `status=complete` | Spike investigation |
| `models_used` missing expected segmentation tag | Alert |

## 4. Rollback switches

| Action | How |
|--------|-----|
| Disable abdominal/chest pipeline only | Stop `abdominal_ct` service or block route at gateway LB |
| Disable cardiac only | Stop `cardiac_ct` |
| Disable spine/neuro only | Stop `spine_neuro` |
| Disable CT Brain only | Stop `ct_brain` or remove route at gateway LB |
| Disable narrative only | Set `CT_*_NARRATIVE_POLICY=off` for that service and redeploy |

Frontends already send explicit `ct_*` modalities; reverting UI to a single CT button is **not** required for backend rollback.

## 5. GPU / VRAM SLO (honest scope)

TotalSegmentator is loaded via upstream APIs. `CT_TOTALSEG_IDLE_EMPTY_CACHE_SEC` **does not** guarantee full weight eviction; it reduces fragmentation and pressure after idle. For strict residency SLOs, measure **process RSS + `nvidia-smi`** in staging with representative series sizes.

## 6. Smoke checklist (post-deploy)

- [ ] `POST /analyze` with `modality=abdominal_ct` returns `ct_product=abdominal_ct`
- [ ] `modality=chest_ct` + `patient_context_json.ct_region=chest_ct` returns `ct_region_context` populated
- [ ] `modality=cardiac_ct` returns `structures.narrative_policy` (default `off`)
- [ ] `modality=spine_ct` reaches `spine_neuro` and returns `ct_product=spine_neuro`
- [ ] Both frontends: each CT subtype button → wizard → upload → report labels match selected subtype
- [ ] `modality=ct_brain` returns `ct_product=ct_brain` and `inference_mode` consistent with env (TorchScript vs `weights_required`)
- [ ] CT Brain idle reaper: set `CT_BRAIN_GPU_IDLE_EMPTY_CACHE_SEC` in staging and confirm logs / VRAM behaviour

## 7. Container smoke (CT Brain cold → warm → idle)

```bash
# From repo root (example; requires compose stack + JWT + sample NCCT)
./scripts/smoke_ct_brain_lifecycle.sh
```

Script expectations: first request may cold-load TorchScript (higher latency); second request warm; after idle threshold, `empty_cache` path runs without crashing.
