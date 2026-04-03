# MRI release, monitoring, and rollback

Companion to `docs/mri_product_contract.md`. Use for staged rollout of **brain MRI** and **spine/neuro MRI** (via `spine_neuro`).

## 1. Environment flags

| Variable | Service | Purpose |
|----------|---------|---------|
| `MRI_NARRATIVE_POLICY` | `02_brain_mri` | `kimi_then_anthropic` (default), `kimi_only`, `anthropic_only`, `off` |
| `MRI_NARRATIVE_VISION` | `02_brain_mri` | `1` (default) = allow Kimi vision with middle slice; `0` = text-only to Kimi |
| `KIMI_MRI_MODEL` | `02_brain_mri` | Moonshot model id (default `moonshot-v1-8k`) |
| `KIMI_API_KEY` / `MOONSHOT_API_KEY` | gateway + services | Kimi auth |
| `ANTHROPIC_API_KEY` | optional fallback | Anthropic narrative when Kimi fails |
| `CT_TOTALSEG_IDLE_EMPTY_CACHE_SEC` | any TotalSeg user | `0` = disabled; `>0` = idle `torch.cuda.empty_cache()` |
| `CT_TOTALSEG_IDLE_CHECK_SEC` | same | Reaper interval (default `30`) |
| `GATEWAY_CORS_ORIGINS` | `gateway` | Comma-separated allowed origins for browser clients; **empty = legacy `*`** (acceptable for local dev only; set explicit origins for public SaaS) |
| `MAX_UPLOAD_MB_BRAIN_MRI` | `02_brain_mri` | Max upload size per request (default `512`) |

API keys: missing keys yield empty narrative (non-fatal) when policy is not `off`.

## 2. Gateway transparency

Brain MRI deep responses include:

- `mri_product`, `gateway_request_modality`, `mri_routing_note`

**Monitoring:** group metrics by `mri_product` and request modality alias, not a single `MRI` bucket.

**Request tracing:** gateway responses include `X-Request-ID` (or echo client `X-Request-ID` / `X-Correlation-ID`). `POST /analyze` forwards this header to downstream services.

## 3. Canary gates (suggested)

For 24–72h after deploy:

| Signal | Gate |
|--------|------|
| HTTP 5xx rate | < 1% of MRI-tagged requests |
| p95 latency | Within agreed SLO (document baseline pre-release) |
| `totalseg_available: false` spike | Investigate weights / GPU / input format |
| Empty `findings` with `status=complete` | Spike investigation |

## 4. Rollback switches

| Action | How |
|--------|-----|
| Disable brain MRI only | Stop `brain_mri` service or block route at LB |
| Disable narrative only | `MRI_NARRATIVE_POLICY=off` + redeploy |
| Disable Kimi vision only | `MRI_NARRATIVE_VISION=0` + redeploy |
| Disable spine/neuro MRI | Stop `spine_neuro` or block route |

## 5. Smoke checklist (post-deploy)

- [ ] `POST /analyze` with `modality=brain_mri` returns `mri_product=brain_mri` and `modality=brain_mri`
- [ ] `modality=mri` (alias) returns same brain pipeline with `mri_routing_note` mentioning alias
- [ ] `modality=spine_mri` reaches `spine_neuro` and returns `modality=spine_neuro`
- [ ] `MRI_NARRATIVE_POLICY=off` yields deterministic stub narrative only
- [ ] `MRI_NARRATIVE_VISION=0` still completes (text-only Kimi path)
- [ ] Production frontend: Brain MRI vs Spine/Neuro MRI entries match contract

## 6. Container smoke (brain MRI cold → warm)

```bash
# Health
curl -s http://localhost:8002/health | jq .

# Analyze (replace with real token and file)
curl -s -X POST http://localhost:8080/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F modality=brain_mri \
  -F file=@/path/to/brain.nii.gz
```

## 7. Incident playbook (short)

1. Set `MRI_NARRATIVE_POLICY=off` if LLM provider is degraded.
2. Scale `brain_mri` replicas or reduce concurrency if GPU OOM.
3. Verify `MODEL_DIR` / TotalSeg cache volume mounted and writable.
