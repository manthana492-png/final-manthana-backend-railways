# Manthana — Final Modal deployment guide (operator runbook)

This document is the **step-by-step** path to go from **Railway gateway (already live)** to **Modal inference** wired in production. It aligns with:

- **Cursor production plan** `vercel-railway-modal_production_2f2ecefa.plan.md` (gateway `*_SERVICE_URL`, retries, timeouts)
- [`MODAL_DEPLOYMENT_PLAN.md`](../../MODAL_DEPLOYMENT_PLAN.md) (architecture, weights, cost)
- Code under [`modal_app/`](.) · [`../shared/`](../shared) · [`../services/`](../services)

---

## 0. What Modal does vs Railway

| Tier | Platform | Role |
|------|----------|------|
| Browser | Vercel | Next.js → `/api/oracle-backend/*` → **`ORACLE_INTERNAL_URL`** (Railway **gateway**) |
| Orchestration | **Railway** | **Gateway** (JWT, `/analyze`, forwards multipart to backends, `/v1/*` → Oracle when deployed) |
| Inference | **Modal** | One **web app** per modality (GPU or CPU), **scale-to-zero**, **Volume** at `/models` |

The **gateway calls Modal over HTTPS** (outbound). You do **not** open a special Railway port “for Modal.” You set **`https://…/analyze/<modality>`** on the gateway.

---

## 1. Machine layout (required before any `modal deploy`)

Modal builds read **your local disk** (or CI checkout). From repo root **`this_studio`** you must have:

| Path | Purpose |
|------|---------|
| `config/cloud_inference.yaml` | LLM / narrative policy SSOT |
| `packages/manthana-inference/` | Editable install in Modal images (`with_manthana_llm_stack` in [`common.py`](./common.py)) |
| `new manthana radiology/manthana-backend/` | `modal_app/`, `services/*`, `shared/` |

**Working directory for all CLI commands below:**  
`new manthana radiology/manthana-backend` (i.e. **manthana-backend root**).

If you only cloned/pushed `manthana-backend` without `packages/` and `config/`, **`modal deploy` will fail** — mirror the full `this_studio` tree on the machine that runs Modal CLI.

---

## 2. One-time Modal account setup

1. Create a [Modal](https://modal.com) account and workspace.
2. Install CLI deps from **manthana-backend**:

   ```bash
   cd "new manthana radiology/manthana-backend"   # adjust to your path
   pip install -r modal_requirements.txt
   ```

3. Authenticate:

   ```bash
   modal token new
   ```

---

## 3. Modal Secret (`manthana-env`)

Create a secret in the Modal dashboard named **`manthana-env`** (override with env **`MANTHANA_MODAL_SECRET`** only if you rename it in code).

**Full production template (recommended):** copy [`modal_app/manthana-modal-secret.env.example`](./manthana-modal-secret.env.example) to `modal_app/manthana-modal-secret.env`, fill secrets and weight paths, then:

```bash
cd "new manthana radiology/manthana-backend"
python -m modal secret create manthana-env --force --from-dotenv "modal_app/manthana-modal-secret.env"
```

(Windows CMD: use `copy` to clone the `.example` file, then the same `python -m modal secret create ...` with your full path in quotes.)

**Minimum recommended keys:**

| Key | Notes |
|-----|--------|
| `OPENROUTER_API_KEY` | Narrative / LLM paths inside services |
| `MODEL_DIR=/models` | Matches Volume mount in deploy stubs |
| `MANTHANA_LLM_REPO_ROOT=/app` | Set in deploy functions; secret can mirror if services read it |

**Per-modality / proprietary weights** (examples — see `docker-compose.yml`, `MODAL_DEPLOYMENT_PLAN.md`, and service READMEs):

- CT brain: `CT_BRAIN_TORCHSCRIPT_PATH`, optional subtype/segmentation paths  
- Brain MRI: `WMH_MODEL_PATH`, `BRAIN_LESION_MODEL_PATH`, `SYNTHSEG_SCRIPT` as needed  
- Oral: weights on Volume — [`../services/14_oral_cancer/WEIGHTS.md`](../services/14_oral_cancer/WEIGHTS.md)  
- Add any other vars your `cloud_inference.yaml` and pipelines expect  

---

## 4. Modal Volume (persistent weights)

- Default name: **`manthana-model-weights`** (`MANTHANA_MODAL_VOLUME` to override).  
- Created on first deploy / volume use (`common.py` → `models_volume()`).  
- Mounted at **`/models`** in GPU apps that declare `volumes={"/models": models_volume()}`.

**Bootstrap TotalSegmentator / heavy tasks (once):**

```bash
cd "new manthana radiology/manthana-backend"
modal run modal_app/bootstrap_weights.py
```

**Upload proprietary artifacts (examples):**

```bash
modal volume put manthana-model-weights ./weights/ich_main.pt ct_brain/ich_main.pt
modal volume put manthana-model-weights ./path/to/oral_effnet_v2m.pt oral_effnet_v2m.pt
```

**Optional — MONAI bundles + VISTA-3D on the same volume**

```bash
set MANTHANA_BOOTSTRAP=monai
modal run modal_app/bootstrap_weights.py
set MANTHANA_BOOTSTRAP=vista
modal run modal_app/bootstrap_weights.py
```

(On macOS/Linux: `MANTHANA_BOOTSTRAP=monai modal run modal_app/bootstrap_weights.py`.)

See [`docs/MODAL_PHASE1_OPERATOR_SETUP.md`](../docs/MODAL_PHASE1_OPERATOR_SETUP.md).

---

## 5. Deploy apps (one Modal app per modality)

Run from **manthana-backend** root. Each file defines `modal.App("<name>")` and `@modal.asgi_app()` → same FastAPI routes as Docker: **`/health`**, **`/analyze/<modality>`**.

### 5.1 GPU services

| Deploy file | Modal app name | Set on Railway gateway (full URL must end with path) |
|-------------|----------------|--------------------------------------------------------|
| [`deploy_ct_brain.py`](./deploy_ct_brain.py) | `manthana-ct-brain` | `CT_BRAIN_SERVICE_URL` → `…/analyze/ct_brain` |
| [`deploy_brain_mri.py`](./deploy_brain_mri.py) | `manthana-brain-mri` | `BRAIN_MRI_SERVICE_URL` → `…/analyze/brain_mri` |
| [`deploy_cardiac_ct.py`](./deploy_cardiac_ct.py) | `manthana-cardiac-ct` | `CARDIAC_CT_SERVICE_URL` → `…/analyze/cardiac_ct` |
| [`deploy_spine_neuro.py`](./deploy_spine_neuro.py) | `manthana-spine-neuro` | `SPINE_NEURO_SERVICE_URL` → `…/analyze/spine_neuro` |
| [`deploy_abdominal_ct.py`](./deploy_abdominal_ct.py) | `manthana-abdominal-ct` | `ABDOMINAL_CT_SERVICE_URL` → `…/analyze/abdominal_ct` |
| [`deploy_body_xray.py`](./deploy_body_xray.py) | `manthana-body-xray` | `XRAY_SERVICE_URL` → `…/analyze/xray` |
| [`deploy_ultrasound.py`](./deploy_ultrasound.py) | `manthana-ultrasound` | `ULTRASOUND_SERVICE_URL` → `…/analyze/ultrasound` |
| [`deploy_pathology.py`](./deploy_pathology.py) | `manthana-pathology` | `PATHOLOGY_SERVICE_URL` → `…/analyze/pathology` |
| [`deploy_cytology.py`](./deploy_cytology.py) | `manthana-cytology` | `CYTOLOGY_SERVICE_URL` → `…/analyze/cytology` |
| [`deploy_mammography.py`](./deploy_mammography.py) | `manthana-mammography` | `MAMMOGRAPHY_SERVICE_URL` → `…/analyze/mammography` |
| [`deploy_lab_report.py`](./deploy_lab_report.py) | `manthana-lab-report` | `LAB_REPORT_SERVICE_URL` → `…/analyze/lab_report` |
| [`deploy_oral_cancer.py`](./deploy_oral_cancer.py) | `manthana-oral-cancer` | `ORAL_CANCER_SERVICE_URL` → `…/analyze/oral_cancer` (**production default: GPU**) |
| [`deploy_ct_brain_vista.py`](./deploy_ct_brain_vista.py) | `manthana-ct-brain-vista` | `CT_BRAIN_VISTA_SERVICE_URL` → `…/analyze/ct_brain` (**A10G premium; same FastAPI route as standard CT brain**) |

**Deploy command pattern:**

```bash
modal deploy modal_app/deploy_ct_brain.py
# …repeat for each app you need
```

Copy the **HTTPS URL** Modal prints after deploy. Your gateway variable must be the **full URL including** `/analyze/<modality>` (same shape as [`../gateway/router.py`](../gateway/router.py) defaults).

### 5.2 CPU services (scale-to-zero, lower idle cost)

| Deploy file | Modal app name | Gateway env |
|-------------|----------------|-------------|
| [`deploy_ecg.py`](./deploy_ecg.py) | `manthana-ecg` | `ECG_SERVICE_URL` → `…/analyze/ecg` |
| [`deploy_dermatology.py`](./deploy_dermatology.py) | `manthana-dermatology` | `DERMATOLOGY_SERVICE_URL` → `…/analyze/dermatology` |

### 5.3 Oral cancer — GPU vs CPU (choose one URL)

| Deploy file | Modal app name | When to use |
|-------------|----------------|-------------|
| [`deploy_oral_cancer.py`](./deploy_oral_cancer.py) | `manthana-oral-cancer` | **Production default** (GPU) |
| [`deploy_oral_cancer_cpu.py`](./deploy_oral_cancer_cpu.py) | `manthana-oral-cancer-cpu` | Cost-saving alternate only |

Set **`ORAL_CANCER_SERVICE_URL`** to **exactly one** deployed app’s `/analyze/oral_cancer` URL — not both.

---

## 6. Railway gateway — wire Modal URLs

On the **gateway** service (e.g. `manthana-api.*`):

1. For **each** modality you deployed, add/update the matching **`*_SERVICE_URL`**.  
2. Value = **full Modal HTTPS URL** + path **`/analyze/<modality>`** (no trailing slash on the base if your copied URL already includes the path).  
3. **Redeploy** gateway or wait for auto-redeploy.

Reference table (duplicate of §5.1 for paste):  
`XRAY_SERVICE_URL`, `BRAIN_MRI_SERVICE_URL`, `CARDIAC_CT_SERVICE_URL`, `SPINE_NEURO_SERVICE_URL`, `ABDOMINAL_CT_SERVICE_URL`, `CT_BRAIN_SERVICE_URL`, `CT_BRAIN_VISTA_SERVICE_URL`, `PATHOLOGY_SERVICE_URL`, `ULTRASOUND_SERVICE_URL`, `CYTOLOGY_SERVICE_URL`, `MAMMOGRAPHY_SERVICE_URL`, `ECG_SERVICE_URL`, `ORAL_CANCER_SERVICE_URL`, `LAB_REPORT_SERVICE_URL`, `DERMATOLOGY_SERVICE_URL`.

Premium routing: gateway modality `ct_brain_vista` requires header `X-Subscription-Tier` ∈ `pro`, `proplus`, `premium`, `enterprise` (set by the Vercel analyse client for Pro users).

**Do not remove** existing Railway vars: `JWT_SECRET`, `SUPABASE_JWT_SECRET`, `REPORT_ASSEMBLY_URL`, `ORACLE_SERVICE_URL` (when Oracle exists), `GATEWAY_CORS_ORIGINS`, `XRAY_TRIAGE_POLICY=always_deep` for slim gateway.

**Behavior already in code (no extra work):**

- Gateway **502/503 retries** and long **timeout** for Modal cold starts (`../gateway/main.py`).  
- **Modal** is never required for Oracle chat — that uses **`ORACLE_SERVICE_URL`** when Oracle is deployed.

---

## 7. Smoke tests

**Per Modal app (after deploy):**

```bash
curl -sS "https://<MODAL_HOST>/health"
```

**Gateway (public):**

```bash
curl -sS "https://<your-gateway>/health"
```

**End-to-end Labs:** log in on Vercel, run one analyze for a modality whose `*_SERVICE_URL` points at Modal; expect 200 after possible cold start.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `modal deploy` fails copying files | Wrong cwd; missing `this_studio` siblings (`packages/`, `config/`) |
| Build fails in `with_manthana_llm_stack` | `packages/manthana-inference` or `cloud_inference.yaml` path wrong |
| Modal `/health` OK, gateway 502 | Wrong `*_SERVICE_URL` (typo, missing `/analyze/...`), or Modal app asleep — retry (gateway retries 502/503) |
| 401 on `/analyze` | Supabase token / `SUPABASE_JWT_SECRET` on gateway |
| OOM on Modal | See deploy files — some apps note A10G vs T4 (`MODAL_DEPLOYMENT_PLAN.md`) |

---

## 9. Optional: batch deploy script

You can run all GPU deploys in sequence (from manthana-backend):

```bash
for f in modal_app/deploy_ct_brain.py modal_app/deploy_brain_mri.py modal_app/deploy_cardiac_ct.py modal_app/deploy_spine_neuro.py modal_app/deploy_abdominal_ct.py modal_app/deploy_body_xray.py modal_app/deploy_ultrasound.py modal_app/deploy_pathology.py modal_app/deploy_cytology.py modal_app/deploy_mammography.py modal_app/deploy_lab_report.py modal_app/deploy_oral_cancer.py modal_app/deploy_ecg.py modal_app/deploy_dermatology.py modal_app/deploy_ct_brain_vista.py; do
  modal deploy "$f"
done
```

Windows: `.\scripts\modal_deploy_all.ps1` from `manthana-backend`.

Skip **`deploy_oral_cancer_cpu.py`** if you use GPU oral.

---

## 10. Related docs

| Doc | Content |
|-----|---------|
| [`README.md`](./README.md) | Quick deploy list + volume put examples |
| [`../../MODAL_DEPLOYMENT_PLAN.md`](../../MODAL_DEPLOYMENT_PLAN.md) | Deep architecture, weight inventory, cost |
| [`../RAILWAY_ENV_REFERENCE.md`](../RAILWAY_ENV_REFERENCE.md) | Gateway + Railway + Vercel env matrix |
| [`../docs/RAILWAY_MODAL_SERVICE_URLS.md`](../docs/RAILWAY_MODAL_SERVICE_URLS.md) | Modal URL checklist including `CT_BRAIN_VISTA_SERVICE_URL` |
| [`../docs/SMOKE_TEST_MODAL_LAUNCH.md`](../docs/SMOKE_TEST_MODAL_LAUNCH.md) | Operator smoke / E2E checklist |
| [`../NVIDIA_COMPLIANCE_REFERENCE.md`](../NVIDIA_COMPLIANCE_REFERENCE.md) | DPDP / CDSCO / ISO reference |
| [`../gateway/router.py`](../gateway/router.py) | Canonical modality → `*_SERVICE_URL` keys |

---

*Last aligned with codebase: Modal deploy stubs under `modal_app/`, `common.py` image helpers, `gateway/router.py` `SERVICE_MAP`.*
