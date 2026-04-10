# Railway + Modal + Vercel — environment and cost reference

Single place for **Manthana Oracle + Labs** production wiring. Paths are relative to the repo layout under `this_studio/`.

## Architecture (three tiers)

| Tier | Platform | Role |
|------|----------|------|
| Frontend | **Vercel** | Next.js; browser calls same-origin `/api/oracle-backend/*`; server uses `ORACLE_INTERNAL_URL`. |
| Middleware | **Railway** | **Gateway** (public), **oracle-service** (private), **report_assembly** (private). Optional: Redis + queue worker (off for launch). |
| Inference | **Modal** | GPU/CPU per modality; scale-to-zero; `*_SERVICE_URL` on gateway point to Modal HTTPS `…/analyze/<modality>`. |

## 1. Gateway (Railway, public)

**Docker:** build from `manthana-backend/` with [`gateway/Dockerfile.railway`](gateway/Dockerfile.railway) (slim; no PyTorch).  
**Port:** `8000`

### Required

| Variable | Notes |
|----------|--------|
| `JWT_SECRET` | Random string, **≥32 chars** (gateway startup fails otherwise). Still used as legacy JWT fallback. |
| `SUPABASE_JWT_SECRET` | From Supabase → Project Settings → API → JWT Secret. Verifies browser `Authorization: Bearer <access_token>`. |
| `SUPABASE_JWT_ISS` | Optional; e.g. `https://<project-ref>.supabase.co/auth/v1` if you enforce issuer. |
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
| `PATHOLOGY_SERVICE_URL` | `…/analyze/pathology` |
| `ULTRASOUND_SERVICE_URL` | `…/analyze/ultrasound` |
| `CYTOLOGY_SERVICE_URL` | `…/analyze/cytology` |
| `MAMMOGRAPHY_SERVICE_URL` | `…/analyze/mammography` |
| `ECG_SERVICE_URL` | `…/analyze/ecg` |
| `ORAL_CANCER_SERVICE_URL` | `…/analyze/oral_cancer` |
| `LAB_REPORT_SERVICE_URL` | `…/analyze/lab_report` |
| `DERMATOLOGY_SERVICE_URL` | `…/analyze/dermatology` |

**Premium:** For modality `ct_brain_vista`, the Vercel client sends `X-Subscription-Tier` (`pro`, `proplus`, `premium`, `enterprise`). The gateway returns **403** if the tier is not allowed.

### Optional

| Variable | Default / notes |
|----------|-----------------|
| `OPENROUTER_API_KEY` | Copilot / LLM paths on gateway if used. |
| `USE_REDIS_QUEUE` | Omit or `0` for launch (sync `/analyze` only). |
| `REDIS_URL` | Only if queue enabled. |
| `PACS_BRIDGE_URL` | Default `http://pacs_bridge:8030`; skip PACS service at launch if unused. |
| `UPLOAD_DIR`, `PDF_OUTPUT_DIR` | Temp dirs for heatmaps/reports. |
| `GATEWAY_PORT` | Default `8000`. |

### Gateway routes added for single-hostname Vercel

- **`/v1/{path}`** → proxied to `ORACLE_SERVICE_URL` (streaming SSE for chat). Requires same Bearer JWT as `/analyze`.

---

## 2. Oracle service (Railway, private)

**Docker:** from `this_studio/` root per [`oracle-2/services/oracle-service/Dockerfile`](../../oracle-2/services/oracle-service/Dockerfile).  
**Port:** `8000` (container)

| Variable | Notes |
|----------|--------|
| `OPENROUTER_API_KEY` | Required for chat/M5 (or `ORACLE_OPENROUTER_API_KEY`). |
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
