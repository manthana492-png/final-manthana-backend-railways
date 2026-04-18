# Railway + Modal + Vercel — environment and cost reference

Single place for **Manthana Oracle + Labs** production wiring. Paths are relative to the repo layout under `this_studio/`.

## Architecture (three tiers)

| Tier | Platform | Role |
|------|----------|------|
| Frontend | **Vercel** | Next.js; browser calls same-origin `/api/oracle-backend/*`; server uses `ORACLE_INTERNAL_URL`. |
| Middleware | **Railway** | **Gateway** (public), **oracle-service** (private), **report_assembly** (private). Optional: Redis + queue worker (off for launch). |
| Inference | **Modal** | GPU/CPU per modality; scale-to-zero; `*_SERVICE_URL` on gateway point to Modal HTTPS `…/analyze/<modality>`. |

## 1. Gateway (Railway, public)

**Docker:** build from **`this_studio/`** (repo root) with [`gateway/Dockerfile.railway`](gateway/Dockerfile.railway). The image bundles `packages/manthana-inference` and `config/cloud_inference.yaml` so `/ai/detect-modality` and related routes work (same pattern as report_assembly).

```bash
cd this_studio
docker build -f "new manthana radiology/manthana-backend/gateway/Dockerfile.railway" -t manthana-gateway .
```

**Railway dashboard (required for this Dockerfile):** set **Root Directory** to the **monorepo root** (the folder that contains `packages/`, `config/`, and `new manthana radiology/`). Do **not** set Root Directory to `new manthana radiology/manthana-backend` — that limits the Docker build context and the image cannot see `packages/manthana-inference` or `config/cloud_inference.yaml`, so the build fails. Set **Dockerfile path** to `new manthana radiology/manthana-backend/gateway/Dockerfile.railway`.

**Port:** `8000`

### Required

| Variable | Notes |
|----------|--------|
| `JWT_SECRET` | Random string, **≥32 chars** (gateway startup fails otherwise). Still used as legacy JWT fallback. |
| `SUPABASE_JWT_SECRET` | From Supabase → Project Settings → API → JWT Secret. Verifies **HS256** browser `Authorization: Bearer <access_token>` when the project uses the legacy shared secret for signing. |
| `SUPABASE_JWT_ISS` | Optional; e.g. `https://<project-ref>.supabase.co/auth/v1` if you enforce issuer (recommended with JWKS verification). |
| `SUPABASE_URL` | Optional; `https://<project-ref>.supabase.co`. When set, the gateway loads **JWKS** and verifies **RS256/ES256** (etc.) access tokens per [Supabase JWT Signing Keys](https://supabase.com/docs/guides/auth/signing-keys). Use with or without `SUPABASE_JWT_SECRET` depending on whether tokens are asymmetric or HS256. |
| `SUPABASE_JWKS_URL` | Optional override; default is `{SUPABASE_JWT_ISS or SUPABASE_URL}/auth/v1/.well-known/jwks.json` (issuer path) or derived from `SUPABASE_URL`. |
| `GATEWAY_CORS_ORIGINS` | Comma-separated origins, e.g. `https://manthana.quaasx108.com`. Empty = `*` (avoid in prod with credentials). |
| `XRAY_TRIAGE_POLICY` | **`always_deep`** for slim Railway image (no torchxrayvision on gateway). |

### Inter-service (use private networking on Railway)

| Variable | Example (private) |
|----------|-------------------|
| `ORACLE_SERVICE_URL` | `http://oracle-service.railway.internal:8000` |
| `REPORT_ASSEMBLY_URL` | `http://report-assembly.railway.internal:8020` |

Docker Compose default without Railway: `http://oracle_service:8000`, `http://report_assembly:8020`.

### Modal inference (full URL including path)

Set each used modality to the Modal deployment URL (from `modal deploy`):

| Variable | Typical path suffix |
|----------|---------------------|
| `XRAY_SERVICE_URL` | `…/analyze/xray` |
| `BRAIN_MRI_SERVICE_URL` | `…/analyze/brain_mri` |
| `CARDIAC_CT_SERVICE_URL` | `…/analyze/cardiac_ct` |
| `SPINE_NEURO_SERVICE_URL` | `…/analyze/spine_neuro` |
| `ABDOMINAL_CT_SERVICE_URL` | `…/analyze/abdominal_ct` |
| `CT_BRAIN_SERVICE_URL` | `…/analyze/ct_brain` |
| `CT_BRAIN_VISTA_SERVICE_URL` | `…/analyze/ct_brain` on the **VISTA-3D** Modal host (`manthana-ct-brain-vista`); same path as standard CT brain |
| `PREMIUM_CT_SERVICE_URL` | `…/analyze/premium_ct` on the **premium unified VISTA** Modal host (`manthana-premium-ct`) |
| `PATHOLOGY_SERVICE_URL` | `…/analyze/pathology` |
| `ULTRASOUND_SERVICE_URL` | `…/analyze/ultrasound` |
| `CYTOLOGY_SERVICE_URL` | `…/analyze/cytology` |
| `MAMMOGRAPHY_SERVICE_URL` | `…/analyze/mammography` |
| `ECG_SERVICE_URL` | `…/analyze/ecg` |
| `ORAL_CANCER_SERVICE_URL` | `…/analyze/oral_cancer` |
| `LAB_REPORT_SERVICE_URL` | `…/analyze/lab_report` |
| `DERMATOLOGY_SERVICE_URL` | `…/analyze/dermatology` |

**Premium:**  
- `ct_brain_vista` accepts `X-Subscription-Tier` in (`pro`, `proplus`, `premium`, `enterprise`).  
- `premium_ct_unified` accepts only (`premium`, `enterprise`) and returns **403** otherwise.

### Optional

| Variable | Default / notes |
|----------|-----------------|
| `OPENROUTER_API_KEY` | **Required** for 95-modality AI orchestration (`POST /ai/detect-modality`, `/ai/interrogate`, `/ai/interpret`) and CoPilot; uses `shared/llm_router.py` + OpenRouter. Also set optional `OPENROUTER_API_KEY_2` for failover. |
| `CLOUD_INFERENCE_CONFIG_PATH` | Optional override; Dockerfile sets `/app/config/cloud_inference.yaml` (baked in). |
| `MANTHANA_LLM_REPO_ROOT` | Optional override; Dockerfile sets `/app`. |
| `USE_REDIS_QUEUE` | Omit or `0` for launch (sync `/analyze` only). |
| `REDIS_URL` | Only if queue enabled. |
| `PACS_BRIDGE_URL` | Default `http://pacs_bridge:8030`; skip PACS service at launch if unused. |
| `UPLOAD_DIR`, `PDF_OUTPUT_DIR` | Temp dirs for heatmaps/reports. |
| `GATEWAY_PORT` | Default `8000`. |
| `MAX_AI_REQUEST_BYTES` | Default `10485760` (10MB). For heavy DICOM-in-JSON uploads consider `20971520` (20MB). |
| `AI_SESSION_TTL_SECONDS` | Default `300`. In-memory interrogation session lifetime. |
| `AI_INTERROGATE_RATE_LIMIT_FREE` / `AI_INTERROGATE_RATE_LIMIT_PAID` / `AI_RATE_WINDOW_SECONDS` | Per-user `/ai/interrogate` limits (single-replica). |
| `ORCH_ALLOWED_GROUPS` | **Production launch (Phase C):** `reports,cardiac_functional,xray,ophthalmology_dental,oncology,ultrasound,pathology,specialized,nuclear,mri,ct`. Empty = allow all (convenient for dev only). |
| `AUDIT_LOG_PATH` | Optional path for JSON-line audit log (default `audit.log` in process cwd; ephemeral on Railway). |
| `AI_ORCH_INTERPRET_WEB_SEARCH` | `always` (default, Labs): allow OpenRouter `web_search` on the first eligible OpenRouter step in the interpreter chain (at most once per `/ai/interpret`). `paid_only`: only when tier is not `free`/`trial`/empty. `never` / `false`: no web search tool. |
| `ORCH_OPENROUTER_TIMEOUT_S` | Optional; HTTP client timeout in seconds for OpenRouter orchestration calls (default `120`). |
| `ORCH_NIM_TIMEOUT_S` | Optional; HTTP client timeout in seconds for NIM orchestration calls (default `120`). |
| `NVIDIA_NIM_API_KEY` | When unset, all `provider: nim` roles in `orch_chains` are skipped automatically; interrogate/detect/interpret still use OpenRouter steps. Validate NIM model slugs with `GET https://integrate.api.nvidia.com/v1/models` before relying on vision/CXR roles. **Phase 2 (optional):** dedicated OCR via NVIDIA `nemoretriever-parse` (`/v1/parse`) is not wired in Phase 1; reports use gateway PDF/text extraction plus `interpreter_reports` / `orch_chains`. |

**Frontend (Vercel):** set `NEXT_PUBLIC_ORCH_PHASE` to `C` for full picker (default in app code), or `A`/`B` for staged rollout; values must match `ORCH_ALLOWED_GROUPS` on the gateway.

### Gateway routes added for single-hostname Vercel

- **`/v1/{path}`** → proxied to `ORACLE_SERVICE_URL` (streaming SSE for chat). Requires same Bearer JWT as `/analyze`.

---

## 2. Oracle service (Railway, private)

**Docker:** from `this_studio/` root per [`oracle-2/services/oracle-service/Dockerfile`](../../oracle-2/services/oracle-service/Dockerfile).  
**Port:** `8000` (container)

| Variable | Notes |
|----------|--------|
| `OPENROUTER_API_KEY` | Required for chat/M5 (or `ORACLE_OPENROUTER_API_KEY`). **Still required** when using free inference: OpenRouter authenticates every request; set `model` to `openrouter/free` via env below (see [Free Models Router](https://openrouter.ai/docs/guides/routing/routers/free-models-router)). |
| `ORACLE_USE_FREE_MODELS` | Set `true` to use roles `oracle_chat_free` / `oracle_m5_free` → primary model **`openrouter/free`** (auto-picks a capable free model) with YAML fallbacks. Default `false` uses paid-tier primaries (`moonshotai/kimi-k2.5:online`, etc.). |
| `CLOUD_INFERENCE_CONFIG_PATH` | Baked in image as `/app/config/cloud_inference.yaml` when using oracle Dockerfile. |
| `FRONTEND_URL` / CORS-related | Set production app origin for CORS if clients hit Oracle directly; behind gateway proxy often minimal. |
| `ORACLE_REDIS_URL` | **Empty for launch** if you want Railway **serverless** sleep (no persistent Redis chatter). |

---

## 3. Report assembly (Railway, private)

**Docker:** from `this_studio/` root:

```bash
cd this_studio
docker build -f "new manthana radiology/manthana-backend/services/report_assembly/Dockerfile.railway" -t manthana-report-assembly .
```

**Port:** `8020`

| Variable | Notes |
|----------|--------|
| `OPENROUTER_API_KEY` | Required (via `shared/llm_router.py`). |
| `MANTHANA_LLM_REPO_ROOT` | Set to `/app` in `Dockerfile.railway`. |
| `CLOUD_INFERENCE_CONFIG_PATH` | `/app/config/cloud_inference.yaml` in image. |

---

## 4. Modal secret (`manthana-env`)

Create in Modal; reference in each `modal deploy` file. Typical keys (see [`modal_app/README.md`](modal_app/README.md) and `MODAL_DEPLOYMENT_PLAN.md`):

- `OPENROUTER_API_KEY`, optional `OPENROUTER_API_KEY_2`
- `MODEL_DIR=/models` (volume mount)
- Per-modality weight paths (`CT_BRAIN_*`, `WMH_MODEL_PATH`, etc.) as in `docker-compose.yml`

---

## 5. Vercel (frontend)

See [`oracle-2/frontend-manthana/manthana/.env.example`](../../oracle-2/frontend-manthana/manthana/.env.example) and [`docs/VERCEL_FIRST_DEPLOY.md`](../../oracle-2/frontend-manthana/manthana/docs/VERCEL_FIRST_DEPLOY.md).

| Variable | Role |
|----------|------|
| `ORACLE_INTERNAL_URL` | **Server-only** HTTPS base of **Railway gateway** (same host for Labs + Oracle via `/v1/*` proxy). |
| `NEXT_PUBLIC_GATEWAY_URL` / `NEXT_PUBLIC_*_API_URL` | Usually `/api/oracle-backend` (same-origin). |
| `NEXT_PUBLIC_SUPABASE_*`, `SUPABASE_SERVICE_ROLE_KEY` | Auth + server features. |
| `NEXT_PUBLIC_APP_URL` | Production site URL for redirects/email. |

---

## 6. Railway cost optimization (dashboard + env)

Official docs: [Pricing](https://docs.railway.com/pricing), [Cost control](https://docs.railway.com/pricing/cost-control), [Serverless](https://docs.railway.com/deployments/serverless), [Private networking](https://docs.railway.com/networking/private-networking).

| Action | Where | Suggestion |
|--------|--------|--------------|
| **Serverless** | oracle-service, report-assembly | Enable so idle services sleep after ~10 min no outbound traffic. Accept cold start on first request (~1–3s). |
| **Keep gateway always-on** | gateway service | Do **not** enable serverless on gateway; it is the only public Railway entry from Vercel. |
| **Private URLs** | Gateway env | `ORACLE_SERVICE_URL`, `REPORT_ASSEMBLY_URL` → `http://<service>.railway.internal:<port>` (no egress). |
| **Replica limits** | Each service | e.g. 0.5 vCPU, 512 MB gateway; tune if OOM. |
| **Usage limits** | Workspace → Usage | Soft alert ~$10; hard limit ~$20 to cap spend. |
| **Skip at launch** | — | No Redis plugin / queue worker / PACS if unused (`USE_REDIS_QUEUE=0`). |

**Note:** Hobby **$5/mo** is a subscription that includes **$5 usage credit**. Three services often exceed $5/mo if gateway runs 24/7; serverless on oracle + report reduces burn. Monitor **Workspace → Usage** after first week.

---

## 7. Smoke checks

1. **Gateway:** `GET https://<gateway>/health` → 200.  
2. **Oracle via gateway:** `GET https://<gateway>/v1/health` with `Authorization: Bearer <supabase_access_token>` → 200.  
3. **Modal:** `curl https://<modal>/health` per deployed app.  
4. **Report assembly:** from gateway, run Labs flow through `POST /report` after a successful analyze.

---

## 8. Related docs

- [`MODAL_DEPLOYMENT_PLAN.md`](../MODAL_DEPLOYMENT_PLAN.md) — full Modal + Railway + Vercel plan (§11 Vercel env).  
- [`manthana-backend/.env.example`](.env.example) — local/backend env hints.
