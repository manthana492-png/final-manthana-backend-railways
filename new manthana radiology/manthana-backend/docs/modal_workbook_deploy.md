# Modal deployment workbook (session log)

This workbook is a **living log** of Modal deploy work: commands run, failures, fixes, and what Modal built or pulled. It complements the full operator runbook: [`../modal_app/MODAL_DEPLOYMENT_FINAL.md`](../modal_app/MODAL_DEPLOYMENT_FINAL.md).

**Project rule:** Whenever you (or Cursor) touch Modal deploy, bootstrap, or image logs, **append a new entry at the top of the [Changelog](#changelog-session-log)** with commands, issue, resolution, and Modal-side details (build steps, `pip` installs, volumes, app URLs). Prefer updating this file in the **same** change or session that fixed the problem.

**Deploy order (production):** Finish **everything on Modal** (secret, volume + weights, all app deploys, `/health` smoke tests, saved HTTPS base URLs) **before** setting **`*_SERVICE_URL`** on Railway. Railway is only wiring; it does not replace an incomplete Modal side.

---

## Modal-first checklist (before Railway)

Work from **`manthana-backend`** on a machine that has the full **`this_studio`** tree (`packages/manthana-inference`, `config/cloud_inference.yaml`, etc.). Use `python -m modal …` if that is how your Python environment exposes the CLI.

| Step | What | Commands / notes |
|------|------|-------------------|
| **A. CLI** | Auth + deps | `pip install -r modal_requirements.txt` then `modal token new` (or refresh token) if needed. |
| **B. Secret** | Production env on Modal | Copy `modal_app/manthana-modal-secret.env.example` → `modal_app/manthana-modal-secret.env`, fill keys and weight paths, then `python -m modal secret create manthana-env --force --from-dotenv "modal_app/manthana-modal-secret.env"`. See [`../modal_app/MODAL_DEPLOYMENT_FINAL.md`](../modal_app/MODAL_DEPLOYMENT_FINAL.md) §3. |
| **C. Volume** | Weights at `/models` | One-time: `python -m modal run modal_app/bootstrap_weights.py`. Upload proprietary `.pt` files with `python -m modal volume put manthana-model-weights …` as in the runbook §4. Optional MONAI/VISTA: set `MANTHANA_BOOTSTRAP` and re-run bootstrap per runbook. |
| **D. Deploy all apps** | Every stub you need in prod | **Batch (PowerShell):** from `manthana-backend`, `.\scripts\modal_deploy_all.ps1` — deploys **15** apps (GPU + CPU + `ct_brain_vista`; **not** `deploy_oral_cancer_cpu.py`). **Or** one-by-one: `python -m modal deploy modal_app\deploy_<name>.py`. Table of app ↔ env var: [`../modal_app/MODAL_DEPLOYMENT_FINAL.md`](../modal_app/MODAL_DEPLOYMENT_FINAL.md) §5 and [`RAILWAY_MODAL_SERVICE_URLS.md`](./RAILWAY_MODAL_SERVICE_URLS.md) (for path suffixes only; set vars on Railway **after** Modal is done). **Plan:** if deploy fails with **“reached limit of N web endpoints”**, see [Workspace web endpoint limit](#workspace-web-endpoint-limit). |
| **E. Smoke** | Each deployed host | `curl -sS "https://<workspace>--<app>-serve.modal.run/health"` — expect OK JSON. Track each printed `https://…serve.modal.run` for later `…/analyze/<modality>`. See [`SMOKE_TEST_MODAL_LAUNCH.md`](./SMOKE_TEST_MODAL_LAUNCH.md). |
| **F. Oral CPU (optional)** | Only if you want CPU oral instead of GPU | `python -m modal deploy modal_app\deploy_oral_cancer_cpu.py` — then point **`ORAL_CANCER_SERVICE_URL`** at **one** oral app only (GPU **or** CPU), never both. |
| **→ Railway** | After A–E pass | Set gateway `*_SERVICE_URL` values and redeploy gateway — **later step**, not part of Modal readiness. |

**After each successful deploy:** run **`curl -sS "https://…-serve.modal.run/health"`** for that app, then the next `python -m modal deploy modal_app\deploy_<next>.py` in runbook §5 order (or `.\scripts\modal_deploy_all.ps1`).

---

## Long “silent” image steps (not stuck)

Modal’s terminal sometimes **stops printing** while a **`RUN python -c …`** layer executes on a remote builder. **No new lines for 5–20+ minutes is normal** when that command downloads large weights.

| App / image | Slow step | Why |
|-------------|-----------|-----|
| **Pathology** (`service_image_pathology`) | `timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, …)` | **timm** downloads **ConvNeXt ImageNet** weights (large file). **Not** Hugging Face `HF_TOKEN`; wait for **`timm ConvNeXt fallback weights cached OK`**. |
| **Cytology** | Same timm warm-up in `service_image_cytology` | Same as pathology. |
| **Ultrasound** | `AutoModel.from_pretrained('microsoft/rad-dino', …)` | HF Hub download; first build can be slow. Optional **`HF_TOKEN`** in secret for rate limits only. |

**What to do:** Wait until the step prints **success** or **fails with an error**. For more detail, open **Modal dashboard → app → deployment → build logs**. If a step exceeds **~45–60 minutes** with no progress, cancel once and retry (transient network).

---

## Workspace web endpoint limit

Some Modal plans cap how many **web endpoints** (FastAPI / `@modal.asgi_app` **serve** functions) you can have **deployed at once** across the workspace. If deploy finishes image build but fails with:

`Deployment failed: reached limit of 8 web endpoints (# already deployed => 8, # in this app => 1)`

then the workspace is **full**: you must **free a slot** or **upgrade**.

| Option | What to do |
|--------|------------|
| **Upgrade** | [Modal plans](https://modal.com/settings) for workspace `manthana492` — higher tiers allow more web endpoints. |
| **Free a slot** | Dashboard → **Apps** → open an app you can pause → **Stop** or **delete** that deployment so it no longer counts. Repeat until you are under the limit, then **`python -m modal deploy modal_app\deploy_<name>.py`** again. |
| **Prioritize** | Keep only the Modalities you need live; deploy **lab_report**, **oral_cancer**, **dermatology**, etc. after removing dev/staging duplicates or old apps. |
| **Second workspace** | Starter plan: add workspace **`manthana492-prod-2`**, CLI profile **`manthana492-prod-2`** (`python -m modal token set` … then `python -m modal profile activate manthana492-prod-2`), recreate secret + volume, deploy overflow apps there — URLs use **`manthana492-prod-2--…`** (see changelog). |

Each Manthana `modal_app/deploy_*.py` app is typically **one** web endpoint. **`modal_deploy_all.ps1`** targets **15** apps — you need a plan (or rotation strategy) that allows that many **concurrent** deployed web apps.

### Second workspace (CMD) — setup first, weights later

Use a **second Modal workspace** (e.g. `manthana492-prod-2`) on the same Starter plan: **8 more web endpoints**; billing credits are **account-level** (same pool as workspace 1). Secrets and volumes are **per workspace** — you must recreate them in workspace 2.

**1) Browser (once):** [Modal → Settings → Tokens](https://modal.com/settings/tokens) → create an API token scoped to workspace **`manthana492-prod-2`**, then add it to your machine (Modal’s token wizard or paste into config). See [Workspaces](https://modal.com/docs/guide/workspaces) → *Create a token for a Workspace*.

**2) CMD — Phase A: point CLI at workspace 2, secret, empty volume (no bootstrap / no model downloads):**

```text
cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"

python -m modal profile list
python -m modal profile activate manthana492-prod-2

python -m modal secret create manthana-env --force --from-dotenv "modal_app\manthana-modal-secret.env"

python -m modal volume list
python -m modal volume create manthana-model-weights
```

If `volume create` errors because the volume already exists, skip that line (volume is already there).

**3) CMD — Phase B: deploy “overflow” apps** (URLs will use the **new** workspace slug, not `manthana492` — set Railway `*_SERVICE_URL` to the printed hostnames):

```text
cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"
python -m modal profile activate manthana492-prod-2

python -m modal deploy modal_app\deploy_lab_report.py
python -m modal deploy modal_app\deploy_mammography.py
python -m modal deploy modal_app\deploy_oral_cancer.py
python -m modal deploy modal_app\deploy_dermatology.py
python -m modal deploy modal_app\deploy_ct_brain_vista.py
python -m modal deploy modal_app\deploy_pathology.py
python -m modal deploy modal_app\deploy_cytology.py
```

**4) CMD — Phase C: populate volume (downloads / heavy)** — run **after** Phase A+B when you are ready for TotalSegmentator cache and optional bundles:

```text
cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"
python -m modal profile activate manthana492-prod-2

python -m modal run modal_app\bootstrap_weights.py
```

Optional (separate runs): `set MANTHANA_BOOTSTRAP=monai` then `python -m modal run modal_app\bootstrap_weights.py`; then `set MANTHANA_BOOTSTRAP=vista` and the same `modal run` (see [`../modal_app/bootstrap_weights.py`](../modal_app/bootstrap_weights.py)). Proprietary `.pt` files: `python -m modal volume put manthana-model-weights <local_file> <remote_path>` per [`../modal_app/MODAL_DEPLOYMENT_FINAL.md`](../modal_app/MODAL_DEPLOYMENT_FINAL.md) §4.

**5) Smoke:** `curl -sS "https://<workspace2-slug>--manthana-<app>-serve.modal.run/health"` for each new app.

**Note:** Modal’s docs use **`modal profile activate`** to switch workspaces from the CLI, not `modal workspace activate`.

---

## 0. Changelog entry template (copy for each session)

Paste a new block **above** the latest dated entry (newest first). Use this shape (real markdown, not nested code fences):

1. Heading: `### YYYY-MM-DD — one-line title`
2. **Shell:** CMD | PowerShell | Cursor terminal (one line).
3. **Commands:** a fenced `text` code block with the exact commands you ran.
4. **Issue:** bullets — error text, Modal UI state, or unexpected behavior.
5. **Resolution:** bullets — files/env/dashboard; note if deploy succeeded on retry.
6. **Modal / image / downloads:** bullets — base image if known, notable `pip` lines from Modal build logs, cache hit vs new layer, deployed app name and `https://…` URL when you have it.

---

## Changelog (session log)

### 2026-04-11 — Workspace `manthana492-prod-2`: Dermatology (CPU) deploy + `/health` OK

**Commands:**

```text
python -m modal profile activate manthana492-prod-2
python -m modal deploy modal_app\deploy_dermatology.py
curl -sS "https://manthana492-prod-2--manthana-dermatology-serve.modal.run/health"
```

**Resolution:**

- Deploy **~186 s**; URL: `https://manthana492-prod-2--manthana-dermatology-serve.modal.run` (CPU **`DEVICE=cpu`**).
- **`GET /health` → 200:** `mode` **`openrouter_vision_v1`**, **`openrouter_configured`**: true; **`ham_v2_weights_present`** / **`b4_weights_present`** false until optional local weights on volume — **`resolved_score_mode`** OpenRouter vision is OK for production gate.

**Next (prod-2):** `deploy_ct_brain_vista.py` → `deploy_pathology.py` → `deploy_cytology.py` (long timm build steps possible) → `python -m modal run modal_app\bootstrap_weights.py` when ready to populate **`manthana-model-weights`** on this workspace.

---

### 2026-04-11 — Workspace `manthana492-prod-2`: Oral cancer redeploy + `/health` OK (post `ORAL_V2M_*` config fix)

**Commands:**

```text
python -m modal profile activate manthana492-prod-2
python -m modal deploy modal_app\deploy_oral_cancer.py
curl -sS "https://manthana492-prod-2--manthana-oral-cancer-serve.modal.run/health"
```

**Resolution:**

- Redeploy **~55 s** (image cache hit after `services/14_oral_cancer/config.py` blank-env fix).
- **`GET /health` → 200:** `status` **`ok`**, **`ready`**: true, **`openrouter_configured`**: true, **`oral_cancer_enabled`**: true, **`oral_v2m_num_classes`**: 3. Checkpoint flags **`b3_checkpoint`** / **`effnet_v2m_weights`** / **`uni_head`** false until proprietary weights exist on this workspace’s volume — expected until **`modal volume put`** / bootstrap; service still reports ready for gated paths.

**Next:** See dermatology prod-2 changelog entry (deployed after this).

---

### 2026-04-11 — Oral cancer Modal import: `ValueError` on empty `ORAL_V2M_NUM_CLASSES` in secret

**Issue:**

- Worker crash on `/health`: `ValueError: invalid literal for int() with base 10: ''` in `services/14_oral_cancer/config.py` — Modal secret had **`ORAL_V2M_NUM_CLASSES=`** (empty string). `os.getenv("ORAL_V2M_NUM_CLASSES", "3")` returns **`""`** when the key exists but is blank, so **`int("")`** fails.

**Resolution:**

- **`services/14_oral_cancer/config.py`:** treat blank **`ORAL_V2M_NUM_CLASSES`** / **`ORAL_V2M_BINARY_OPMD_FRACTION`** as unset (defaults **3** and **0.45**).
- **`modal_app/manthana-modal-secret.env.example`:** set explicit **`ORAL_V2M_NUM_CLASSES=3`** and **`ORAL_V2M_BINARY_OPMD_FRACTION=0.45`** so copied secrets do not ship empty values.

**Next:** `python -m modal profile activate manthana492-prod-2` → `python -m modal deploy modal_app\deploy_oral_cancer.py` → `curl` oral-cancer `/health`.

---

### 2026-04-11 — Workspace `manthana492-prod-2`: Mammography deploy OK; `/health` `models_loaded` false until first Mirai load

**Commands:**

```text
python -m modal profile activate manthana492-prod-2
python -m modal deploy modal_app\deploy_mammography.py
curl -sS "https://manthana492-prod-2--manthana-mammography-serve.modal.run/health"
```

**Resolution:**

- Deploy **~245 s**; URL: `https://manthana492-prod-2--manthana-mammography-serve.modal.run`.
- **`GET /health` → 200:**

```json
{"service":"mammography","status":"ok","models_loaded":false,"gpu_available":true}
```

- **`models_loaded: false` is expected on cold `/health`:** `services/12_mammography/main.py` sets **`models_loaded`** from **`inference.is_loaded()`**, which is **true only after Mirai** (`Lab-Rasool/Mirai` via HF **`snapshot_download`**) has been loaded in-process. Unlike some images, **`service_image_mammography()`** does **not** bake Mirai weights at **image build**; Mirai downloads on **first** pipeline path that calls **`_load_mirai()`** (needs **`HF_TOKEN`** in secret if Hub rate-limits). After a successful **`POST /analyze/mammography`**, **`models_loaded`** can flip **true** on subsequent `/health` in the same warm container.
- **Railway:** **`MAMMOGRAPHY_SERVICE_URL`** → this base URL; path **`/analyze/mammography`**.

**Next (prod-2):** `python -m modal deploy modal_app\deploy_oral_cancer.py` then `curl` `…--manthana-oral-cancer-serve.modal.run/health` (exact slug from Modal output).

---

### 2026-04-11 — Workspace `manthana492-prod-2`: Lab report deploy OK + `/health` (overflow from 8-endpoint limit)

**Shell:** CMD.

**Commands:**

```text
python -m modal token set --token-id ak-... --token-secret "..." --profile manthana492-prod-2
python -m modal profile activate manthana492-prod-2
python -m modal secret create manthana-env --force --from-dotenv "modal_app\manthana-modal-secret.env"
python -m modal volume create manthana-model-weights
python -m modal deploy modal_app\deploy_lab_report.py
curl -sS "https://manthana492-prod-2--manthana-lab-report-serve.modal.run/health"
```

**Resolution:**

- Second workspace **`manthana492-prod-2`** on Starter adds **8 more web endpoints** (same billing account / credits). CLI profile name: **`manthana492-prod-2`** (`python -m modal profile list`).
- **`manthana-lab-report`** deployed successfully (**~316 s**); base URL: `https://manthana492-prod-2--manthana-lab-report-serve.modal.run` — gateway **`LAB_REPORT_SERVICE_URL`** must use this host (path **`/analyze/lab_report`** per runbook).
- **Smoke:** `curl -sS "https://manthana492-prod-2--manthana-lab-report-serve.modal.run/health"` — expect JSON with `"service":"lab_report"`; `status` is **`ok`** when OpenRouter key is present in secret, else **`no_api_key`**.
- **`GET /health` verified (200):**

```json
{"service":"lab_report","status":"ok","models_loaded":true,"llm":"openrouter","gpu_available":true,"version":"1.0.0"}
```

**Next (prod-2 profile, same volume):** Mammography deployed (separate changelog entry) — then `python -m modal deploy modal_app\deploy_oral_cancer.py` → `deploy_dermatology.py` → `deploy_ct_brain_vista.py` → `deploy_pathology.py` → `deploy_cytology.py`; then `python -m modal run modal_app\bootstrap_weights.py` when ready to fill this workspace’s volume.

---

### 2026-04-11 — Lab report (`manthana-lab-report`): image OK; deploy blocked (8 web endpoint limit)

**Commands:**

```text
python -m modal deploy modal_app\deploy_lab_report.py
```

**Issue:**

- Image layers built successfully; Modal created mounts and printed `https://manthana492--manthana-lab-report-serve.modal.run`, then: **`Deployment failed: reached limit of 8 web endpoints (# already deployed => 8, # in this app => 1)`**.

**Resolution:**

- **Not a code bug** — workspace **`manthana492`** is at the plan’s **maximum concurrent web endpoints**. Either **upgrade** the Modal plan, or **stop/delete** other deployed apps in the dashboard until at least one slot is free, then re-run deploy.
- **`deploy_oral_cancer.py`** (and any further web app) will hit the **same** limit until a slot is freed or the plan is raised.

**Next:** Free one endpoint or upgrade → `python -m modal deploy modal_app\deploy_lab_report.py` again → `curl` lab-report `/health`.

---

### 2026-04-11 — ECG (`manthana-ecg`) CPU deploy + `/health` OK

**Commands:**

```text
python -m modal deploy modal_app\deploy_ecg.py
curl -sS "https://manthana492--manthana-ecg-serve.modal.run/health"
```

**Resolution:**

- Deploy **~133 s**; URL: `https://manthana492--manthana-ecg-serve.modal.run` (web function **`serve`**).
- **CPU-only** image (`service_image_ecg_cpu`); no Modal weights volume — heuristics + **neurokit2** + OpenRouter **`narrative_ecg`** per `cloud_inference.yaml`.
- **`GET /health` → 200:**

```json
{"service":"ecg","status":"ok","models_loaded":true,"component_health":{"ecg_pipeline_version":"heuristic-prompt-1.0.0","ecg_branch":"heuristic_neurokit2_openrouter","ecg_dl_weights":"none"},"gpu_available":false,"version":"1.0.0"}
```

**Next (light):** When wiring the gateway, set **`ECG_SERVICE_URL`** to the Modal base above (gateway must call **`…/analyze/ecg`**). **Next Modal (same batch order as `modal_deploy_all.ps1`):** `python -m modal deploy modal_app\deploy_dermatology.py` then `curl` `…--manthana-dermatology-serve.modal.run/health`.

---

### 2026-04-11 — Ultrasound (`manthana-ultrasound`) deploy + `/health` OK (Rad-DINO cached)

**Commands:**

```text
python -m modal deploy modal_app\deploy_ultrasound.py
curl -sS "https://manthana492--manthana-ultrasound-serve.modal.run/health"
```

**Resolution:**

- Deploy **~264 s**; URL: `https://manthana492--manthana-ultrasound-serve.modal.run`.
- Build: Rad-DINO prefetch **`Rad-DINO weights cached OK`** (HF Hub may warn about unauthenticated requests — optional **`HF_TOKEN`** in secret for rate limits).
- **`GET /health`:** `{"service":"ultrasound","status":"ok","models_loaded":true,"gpu_available":true}`

**Next (Modal):** `python -m modal deploy modal_app\deploy_pathology.py` then curl `…--manthana-pathology-serve.modal.run/health`.

---

### 2026-04-11 — Ultrasound image build: add `torchvision` for Rad-DINO / `AutoImageProcessor`

**Issue:**

- Modal image build failed on Rad-DINO prefetch: `ImportError: AutoImageProcessor requires the Torchvision library but it was not found` (`service_image_ultrasound` `run_commands`).

**Resolution:**

- **`services/09_ultrasound/requirements.txt`:** add **`torchvision>=0.17`** so `AutoImageProcessor` / Rad-DINO prefetch in `service_image_ultrasound()` has torchvision (Transformers requires it for vision processors).

**Next:** `python -m modal deploy modal_app\deploy_ultrasound.py` (rebuild).

---

### 2026-04-11 — Body X-ray (`manthana-body-xray`) deploy + `/health` OK

**Commands:**

```text
python -m modal deploy modal_app\deploy_body_xray.py
curl -sS "https://manthana492--manthana-body-xray-serve.modal.run/health"
```

**Resolution:**

- Deploy **~306 s**; URL: `https://manthana492--manthana-body-xray-serve.modal.run`.
- **Modal image build:** TorchXRayVision **DenseNet** weights pre-downloaded during image build (`nih-pc-chex-mimic…`, `chex-…`, `mimic_nb-…` → **`TXRV weights pre-download OK`**); then **`ENV XRAY_HEATMAP_URL_MODE=none`**.

**`/health` (200):**

```json
{"service":"body_xray","status":"ok","models_loaded":false,"gpu_available":true,"version":"1.0.0"}
```

**Note:** `services/01_body_xray/main.py` always returns **`"status": "ok"`** for liveness; **`models_loaded`** comes from **`pipeline_chest.is_loaded()`** (whether chest TXRV models are **already loaded in process**). **`false`** on first `/health` after cold start is common if models load lazily on first chest request — not a failed deploy if analyze works.

**Next (Modal):** `python -m modal deploy modal_app\deploy_ultrasound.py` then curl the printed `…-ultrasound-serve.modal.run/health`.

---

### 2026-04-11 — Abdominal CT (`manthana-abdominal-ct`) deploy OK; `/health` degraded (Comp2Comp)

**Commands:**

```text
python -m modal deploy modal_app\deploy_abdominal_ct.py
curl -sS "https://manthana492--manthana-abdominal-ct-serve.modal.run/health"
```

**Resolution:**

- Deploy **~516 s**; URL: `https://manthana492--manthana-abdominal-ct-serve.modal.run`.

**`/health` body (200):**

```json
{"service":"abdominal_ct","status":"degraded","models_loaded":false,"component_health":{"totalseg":true,"comp2comp":false,"sybil":true},"gpu_available":true}
```

**Why `degraded` / `models_loaded: false`:** In `services/08_abdominal_ct/main.py`, **`ok = all(ch.values())`** — every flag in `component_health` must be **true**. Here **`comp2comp`** is **false** (Comp2Comp CLI at `COMP2COMP_DIR`/`C2C` not passing the health check in this container). **TotalSegmentator** and **Sybil** are **true**; the HTTP response is still **200** with an honest degraded signal, not a crash.

**If you need `status: "ok"`:** Fix Comp2Comp in the image (binary present + `C2C --help` exit 0 or 2), or relax health logic in code (only if product policy allows partial readiness).

**Next (Modal):** `python -m modal deploy modal_app\deploy_body_xray.py` then curl `…--manthana-body-xray-serve.modal.run/health`.

---

### 2026-04-11 — Spine neuro (`manthana-spine-neuro`) deploy + `/health` OK

**Commands:**

```text
python -m modal deploy modal_app\deploy_spine_neuro.py
curl -sS "https://manthana492--manthana-spine-neuro-serve.modal.run/health"
```

**Resolution:**

- Deploy finished in **~747.7 s** (~12.5 min); Modal URL: `https://manthana492--manthana-spine-neuro-serve.modal.run`.
- **`GET /health` → 200**, body:

```json
{"service":"spine_neuro","status":"ok","models_loaded":true,"component_health":{"totalseg":true,"comp2comp":false,"ready":true,"full":false},"gpu_available":true}
```

- **Notes:** TotalSegmentator up; Comp2Comp not reported ready (`comp2comp`: false) — optional or not configured in this env; **`ready`: true** for service gate.

**Next (Modal):** `python -m modal deploy modal_app\deploy_abdominal_ct.py` then `curl` the printed `…-abdominal-ct-serve.modal.run/health`.

---

### 2026-04-11 — Cardiac CT (`manthana-cardiac-ct`) Modal deploy succeeded

**Commands:**

```text
cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"
python -m modal deploy modal_app\deploy_cardiac_ct.py
```

**Resolution:** **`✓ App deployed in 404.470s`**.

**Modal / mounts:** `deploy_cardiac_ct.py`, `packages/manthana-inference`, `cloud_inference.yaml`, `shared`, **`modal_app`** (whole dir mount), `services/04_cardiac_ct`.

**Web endpoint:** `https://manthana492--manthana-cardiac-ct-serve.modal.run` — use this **literal hostname** in `curl` (do not paste the placeholder `<that-host>`).

**Note:** `curl: (3) URL rejected: Bad hostname` happens if the URL still contains `<that-host>` or angle brackets.

---

### 2026-04-11 — Brain MRI `/health` 200 OK on Modal (post path + Prima fixes)

**Shell:** CMD.

**Commands:**

```text
curl -sS "https://manthana492--manthana-brain-mri-serve.modal.run/health"
```

**Issue:**

- None (verification after redeploy).

**Resolution:**

- **`GET /health` → 200 OK** (Modal log: ~8.4 s wall, ~2.9 s execution — cold GPU + imports).
- **Sample body:** `{"service":"brain_mri","status":"ok","models_loaded":{"totalseg":true,"synthseg":false,"prima":false,"prima_configured":false,"prima_weights_present":false,"ready":true,"full":false},"gpu_available":true}` — **ready: true** with TotalSegmentator; SynthSeg/Prima optional paths not fully wired in this env (expected unless configured).

**Modal / image / downloads:**

- Endpoint: `https://manthana492--manthana-brain-mri-serve.modal.run/health`.

---

### 2026-04-11 — Brain MRI: `is_prima_available` on `shared/prima_pipeline` (import shadow fix)

**Issue:** `ImportError: cannot import name 'is_prima_available' from 'prima_pipeline' (/app/shared/prima_pipeline.py)` — `inference.py` prepends `shared/` first, so the **shared** module shadows **`services/02_brain_mri/prima_pipeline.py`**.

**Resolution:** Implement **`is_prima_available()`** in **`shared/prima_pipeline.py`** (same config/weights checks as the service wrapper). Redeploy **`deploy_brain_mri.py`**.

---

### 2026-04-11 — Fix `main.py` / `inference.py` `parents[2]` on Modal; brain MRI `max_containers`

**Issue:**

- Worker: `/app/main.py` used `Path(__file__).parents[2]` → **`IndexError`** (only two parents above `main.py` in flat `/app`).
- Health checks appeared to spin **several containers** at once: `deploy_brain_mri` used **`max_containers=3`**, so Modal could run up to three GPU workers in parallel.

**Resolution:**

- New **`shared/manthana_paths.py`** with **`backend_root_from_service_file()`** (respects `MANTHANA_BACKEND_ROOT`, treats paths under **`/app`** as flat image layout).
- **`services/02_brain_mri/main.py`**, **`services/11_ct_brain/main.py`**, **`services/11_ct_brain/inference.py`:** resolve backend root via that helper (import or importlib from `/app/shared/...` or repo `shared/`).
- **`modal_app/deploy_brain_mri.py`:** **`max_containers=1`** so probes do not open three cold GPUs at once (raise later if you need throughput).

---

### 2026-04-11 — Fix `IndexError` on `studio_root()` in Modal worker (`common.py`)

**Issue:**

- Worker import of `deploy_brain_mri.py` failed: `studio_root()` used `Path(__file__).parents[3]`, but in the container `common.py` is `/app/modal_app/common.py` (only three parents above the file → `parents[3]` raises `IndexError`).

**Resolution:**

- **`studio_root()`:** if `(backend_root() / "packages" / "manthana-inference").is_dir()`, return `backend_root()` (flattened `/app` image layout); else keep `parents[3]` for local `this_studio` checkout.
- **`backend_root()`:** optional override `MANTHANA_BACKEND_ROOT`; unchanged default for local, `/app` in worker via `parent.parent` of `common.py`.
- **`copy_shared` env:** also set `MANTHANA_BACKEND_ROOT` and `MANTHANA_STUDIO_ROOT` to `/app` for workers.

---

### 2026-04-10 — Fix `ModuleNotFoundError: modal_app` in Modal workers (`copy_shared`)

**Shell:** (any).

**Issue:**

- Cold start / `serve` import failed: `ModuleNotFoundError: No module named 'modal_app'` from `/root/deploy_brain_mri.py` line 7 (`from modal_app.common import …`). Modal hydrates deploy stubs at `/root/` without the local package layout.

**Resolution:**

- **`modal_app/common.py` — `copy_shared()`:** also `add_local_dir` the whole `modal_app/` tree to **`/app/modal_app`** and set **`PYTHONPATH=/app`** so `modal_app` is importable in the worker. Redeploy affected apps after this change.

---

### 2026-04-10 — Brain MRI (`manthana-brain-mri`) deploy succeeded

**Shell:** CMD (`manthana-backend`).

**Commands:**

```text
cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"
python -m modal deploy modal_app\deploy_brain_mri.py
```

**Issue:**

- None.

**Resolution:**

- **`✓ App deployed in 391.237s`**.

**Modal / image / downloads:**

- **Mounts:** `deploy_brain_mri.py`, `packages\manthana-inference`, `config\cloud_inference.yaml`, `shared`, `services\02_brain_mri`.
- **Late build steps (log):** `COPY . /` image layers `im-dmGrHe7iSOCuYUbHEkNmNj` (~7.5s), `im-Nr4eHsHcVEBF1VsHRrfQsC` (~9.2s); **`ENV MANTHANA_USE_MONAI_CT_LOADER=1`** layer `im-gip9ialJmmf7gygMHf6ud5` (~6s). Earlier long steps reused or rebuilt service-specific stacks on Modal builders (not local disk).
- **Web endpoint:** `https://manthana492--manthana-brain-mri-serve.modal.run`
- **Dashboard:** `https://modal.com/apps/manthana492/main/deployed/manthana-brain-mri`

---

### 2026-04-10 — CT brain (`manthana-ct-brain`) deploy succeeded after `add_local_*` fix

**Shell:** CMD (from `manthana-backend`).

**Commands:**

```text
cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"
python -m modal deploy modal_app\deploy_ct_brain.py
```

**Issue:**

- None for this run (prior run failed on `copy_local_dir`; see next entry).

**Resolution:**

- Deploy completed: **`✓ App deployed in 352.814s`**.

**Modal / image / downloads:**

- **Mounts:** `deploy_ct_brain.py`, `this_studio\packages\manthana-inference`, `this_studio\config\cloud_inference.yaml`, `manthana-backend\shared`, `manthana-backend\services\11_ct_brain`.
- **Image build (remote, via Modal mirror `pypi-mirror.modal.local`):** service `/.requirements.txt` layer — **torch 2.11.0** (~531 MB wheel) plus NVIDIA CUDA stack (e.g. `nvidia-cudnn-cu13`, `nvidia-cublas`, `triton`, etc.), **fastapi**, **numpy**, **scipy**, **nibabel**, **opencv-python-headless**, **pydicom**, **uvicorn**, etc. First major image reported **`Built image im-GXW3LCyahcfZRG4wcCb0qN in 125.09s`**.
- **MONAI layer:** `pip install monai==1.3.0` — **`Built image im-YrOTwSLWpb3KSir4N3R4t0 in 25.22s`**.
- **LLM stack:** `pip install -e /app/packages/manthana-inference` (pulls **openai**, **httpx**, etc.), then **`instructor>=1.0.0`** (pulls **aiohttp**, **rich**, **typer**, etc.); **`jiter`** downgraded for instructor constraint (0.14 → 0.13).
- Further **`COPY . /`** slices and **`ENV MANTHANA_USE_MONAI_CT_LOADER=1`** as defined in `common.py` / deploy stub.
- **Web endpoint:** `https://manthana492--manthana-ct-brain-serve.modal.run` (function name `serve`).
- **Dashboard:** `https://modal.com/apps/manthana492/main/deployed/manthana-ct-brain`.

**Next (Modal-first, before Railway):**

1. **Health check:** `curl -sS "https://manthana492--manthana-ct-brain-serve.modal.run/health"`.
2. **Continue Modal rollout:** deploy remaining apps (`.\scripts\modal_deploy_all.ps1` or per-file `python -m modal deploy modal_app\deploy_….py`) and complete secret / volume / smoke steps in [Modal-first checklist](#modal-first-checklist-before-railway).
3. **Railway later:** only after the checklist is done, set **`CT_BRAIN_SERVICE_URL`** (and siblings) per [`RAILWAY_MODAL_SERVICE_URLS.md`](./RAILWAY_MODAL_SERVICE_URLS.md).

---

### 2026-04-10 — `Image.copy_local_dir` removed in Modal client; workbook + `common.py` migration

**Shell:** Windows (CMD / PowerShell; user Store Python 3.13).

**Commands:**

```text
cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"
python -m modal deploy modal_app\deploy_ct_brain.py
```

**Issue:**

- Deploy failed **at import time** (before remote image build completed): `AttributeError: 'Image' object has no attribute 'copy_local_dir'`.
- Stack: `deploy_ct_brain.py` → `service_image_ct_brain()` → `with_manthana_llm_stack()` in `modal_app/common.py` (`copy_local_dir` / `copy_local_file`).

**Resolution:**

- Replaced all `copy_local_dir` / `copy_local_file` with `add_local_dir` / `add_local_file`, using `remote_path=…` and `copy=True` where the image build must see files (matches old “copy into layer” behavior).
- File: `modal_app/common.py` (all service image builders + `with_manthana_llm_stack` + `copy_shared`).

**Modal / image / downloads:**

- Initial failure happened **before** any remote image build. **Successful remote build and endpoint** are recorded in the **preceding** changelog entry (same date).

**Docs:**

- Created this workbook: `docs/modal_workbook_deploy.md`.

---

## 1. Goal

From **`manthana-backend`** (repo root for backend + Modal stubs), run:

```text
python -m modal deploy modal_app\deploy_ct_brain.py
```

That loads `modal_app/deploy_ct_brain.py`, which builds a `modal.Image` via `modal_app/common.py` (`service_image_ct_brain()`, MONAI stack, shared copy, etc.) and registers the Modal app.

---

## 2. Where to run the command (Windows)

Use **any** terminal whose working directory is **`manthana-backend`** and where **`python`** is the same interpreter that has `modal` installed (often the Store Python 3.13 path you saw in tracebacks).

| Shell | Change directory | Then deploy |
|--------|------------------|-------------|
| **CMD** | `cd /d "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"` | `python -m modal deploy modal_app\deploy_ct_brain.py` |
| **PowerShell** | `Set-Location "d:\studio-backup\this_studio\new manthana radiology\manthana-backend"` | same `python -m modal deploy ...` |

**Note:** In **PowerShell**, `&&` between commands is **not** valid on older versions; run `cd` (or `Set-Location`) and `python -m modal ...` on **separate lines**, or use `;` as a separator if your PowerShell supports it.

**Cursor / VS Code:** Terminal → New Terminal, then `cd` / `Set-Location` to `manthana-backend` as above.

---

## 3. First incident (detail) — deploy-time import

*(Summary also in [Changelog](#changelog-session-log).)*

### Symptom

`modal deploy` failed **while importing** the deploy module, **before** a remote image build finished:

```text
AttributeError: 'Image' object has no attribute 'copy_local_dir'
```

### Stack (abbreviated)

1. `modal_app\deploy_ct_brain.py` — `@app.function(image=service_image_ct_brain(), ...)`
2. `modal_app\common.py` — `service_image_ct_brain()` → `with_manthana_llm_stack(img)`
3. `modal_app\common.py` — line calling **`img.copy_local_dir(...)`** (and nearby **`copy_local_file`**)

### Root cause

The **Modal Python client** you are using (e.g. **1.4.x**) removed the older **`Image.copy_local_dir`** / **`Image.copy_local_file`** APIs. Replacements are:

- **`add_local_dir(local_path, remote_path=..., copy=...)`**
- **`add_local_file(local_path, remote_path, *, copy=...)**

Use **`copy=True`** when the files must exist **inside the image filesystem** for **later build steps** (for example `pip install -e /app/packages/...` or `run_commands` that read those paths). That matches the old “copy into the image layer” behavior.

---

## 4. What we changed (fix)

**File:** `modal_app/common.py`

| Before | After |
|--------|--------|
| `img.copy_local_dir(local, "/app")` | `img.add_local_dir(local, remote_path="/app", copy=True)` |
| `img.copy_local_file(local, "/app/config/...")` | `img.add_local_file(local, "/app/config/...", copy=True)` |
| Same pattern for `packages/manthana-inference`, `shared/`, and each `services/<id>_*` tree | `remote_path=...` + `copy=True` consistently |

After the change, **`grep` for `copy_local_dir` / `copy_local_file`** under `manthana-backend` should return **no** matches.

---

## 5. Related work (same rollout, different files)

These items came up in the same Modal rollout context but are **not** the `copy_local_dir` error:

- **`modal_app/bootstrap_weights.py`:** Avoid importing `modal_app.common` from code that Modal mounts in isolation (can cause **`ModuleNotFoundError: modal_app`**). Prefer local constants / small duplicates where needed.
- **TotalSegmentator:** Some tasks require a **license** and may **`sys.exit(1)`**; bootstrap logic was adjusted so optional tasks and **`SystemExit`** do not derail the whole job when unlicensed (see env examples and bootstrap task lists in repo).

For full secrets, volumes, and deploy order, follow **`modal_app/MODAL_DEPLOYMENT_FINAL.md`**.

---

## 6. After the fix — what you should see

Re-run from **`manthana-backend`**:

```text
python -m modal deploy modal_app\deploy_ct_brain.py
```

**Expected:** No `AttributeError` on `copy_local_dir`. Modal may then proceed to **image build** and **app registration**; any new errors would be **different** (network, missing local paths `packages/` / `config/`, secret names, etc.). Capture the **new** traceback if something else appears.

---

## 7. Quick reference — same pattern for other apps

Other `modal_app/deploy_*.py` stubs that use `modal_app/common.py` image builders **inherit** the same `add_local_dir` / `add_local_file` fix. Redeploy each app you care about after pulling the updated `common.py`:

```text
python -m modal deploy modal_app\deploy_<name>.py
```

---

## 8. Document map

| Doc | Role |
|-----|------|
| This file | **Living changelog** + Windows CLI notes + first `Image` API migration |
| [`modal_app/MODAL_DEPLOYMENT_FINAL.md`](../modal_app/MODAL_DEPLOYMENT_FINAL.md) | **Full** Modal + secret + volume + Railway wiring |
| [`MODAL_PHASE1_OPERATOR_SETUP.md`](./MODAL_PHASE1_OPERATOR_SETUP.md) | Phase-1 operator setup |
| [`RAILWAY_MODAL_SERVICE_URLS.md`](./RAILWAY_MODAL_SERVICE_URLS.md) | Gateway `*_SERVICE_URL` env vars |

---

*Maintainers: add new sessions to **Changelog** at the top; keep section 3+ as reference or fold long stories into changelog only.*
