# Manthana CT/MRI + X-ray + Ultrasound + Pathology + Cytology + Mammography + Lab report + Oral cancer — Modal deployment

GPU inference uses [Modal](https://modal.com): scale-to-zero, persistent `Volume` at `/models`, existing FastAPI apps unchanged.

## Prerequisites

1. **Directory layout** (Modal build reads files from your machine):

   - `this_studio/config/cloud_inference.yaml`
   - `this_studio/packages/manthana-inference/`
   - `this_studio/new manthana radiology/manthana-backend/` (this repo)

2. **Modal**: `pip install -r modal_requirements.txt` from `manthana-backend`, then `modal token new`.

3. **Secret** in Modal dashboard named `manthana-env` (override with `MANTHANA_MODAL_SECRET`), or use the full template: copy [`modal_app/manthana-modal-secret.env.example`](./manthana-modal-secret.env.example) → `modal_app/manthana-modal-secret.env`, fill values, then `python -m modal secret create manthana-env --force --from-dotenv modal_app/manthana-modal-secret.env`. Minimum keys include:

   - `OPENROUTER_API_KEY`
   - `MODEL_DIR=/models`
   - `MANTHANA_LLM_REPO_ROOT=/app` (optional; deploy stubs set this)
   - CT brain: `CT_BRAIN_TORCHSCRIPT_PATH`, optional subtype/segmentation paths
   - Brain MRI: `WMH_MODEL_PATH`, `BRAIN_LESION_MODEL_PATH`, `SYNTHSEG_SCRIPT` if using SynthSeg
   - Narrative policies: same vars as `docker-compose.yml` for each service

4. **Volume** `manthana-model-weights` (override with `MANTHANA_MODAL_VOLUME`) — created automatically on first deploy, or create explicitly.

## Bootstrap TotalSegmentator weights (once)

From `manthana-backend`:

```bash
modal run modal_app/bootstrap_weights.py
```

Then upload proprietary weights, for example:

```bash
modal volume put manthana-model-weights ./weights/ich_main.pt ct_brain/ich_main.pt
```

Oral cancer (clinical photo weights on the same volume):

```bash
modal volume put manthana-model-weights ./path/to/oral_effnet_v2m.pt oral_effnet_v2m.pt
# optional legacy B3 head:
# modal volume put manthana-model-weights ./path/to/oral_cancer_finetuned.pt oral_cancer_finetuned.pt
```

See [`services/14_oral_cancer/WEIGHTS.md`](../services/14_oral_cancer/WEIGHTS.md) for class order, `ORAL_PREFER_V2M`, and binary (2-class) mapping.

## Deploy each GPU service

From `manthana-backend`:

```bash
modal deploy modal_app/deploy_ct_brain.py
modal deploy modal_app/deploy_brain_mri.py
modal deploy modal_app/deploy_cardiac_ct.py
modal deploy modal_app/deploy_spine_neuro.py
modal deploy modal_app/deploy_abdominal_ct.py
modal deploy modal_app/deploy_body_xray.py
modal deploy modal_app/deploy_ultrasound.py
modal deploy modal_app/deploy_pathology.py
modal deploy modal_app/deploy_cytology.py
modal deploy modal_app/deploy_mammography.py
modal deploy modal_app/deploy_lab_report.py
modal deploy modal_app/deploy_oral_cancer.py
```

Copy each HTTPS URL Modal prints. Full analyze URL shape: `https://….modal.run` (Modal serves the ASGI app at `/analyze/...`).

## Railway gateway

Set environment variables on the gateway service:

| Variable | Example |
|----------|---------|
| `CT_BRAIN_SERVICE_URL` | `https://….modal.run/analyze/ct_brain` |
| `BRAIN_MRI_SERVICE_URL` | `https://….modal.run/analyze/brain_mri` |
| `CARDIAC_CT_SERVICE_URL` | `https://….modal.run/analyze/cardiac_ct` |
| `SPINE_NEURO_SERVICE_URL` | `https://….modal.run/analyze/spine_neuro` |
| `ABDOMINAL_CT_SERVICE_URL` | `https://….modal.run/analyze/abdominal_ct` |
| `XRAY_SERVICE_URL` | `https://….modal.run/analyze/xray` |
| `ULTRASOUND_SERVICE_URL` | `https://….modal.run/analyze/ultrasound` |
| `ECG_SERVICE_URL` | **Modal CPU** `https://….modal.run/analyze/ecg` (recommended, no volume) or Railway `…/analyze/ecg` |
| `PATHOLOGY_SERVICE_URL` | `https://….modal.run/analyze/pathology` |
| `CYTOLOGY_SERVICE_URL` | `https://….modal.run/analyze/cytology` |
| `MAMMOGRAPHY_SERVICE_URL` | `https://….modal.run/analyze/mammography` |
| `LAB_REPORT_SERVICE_URL` | `https://….modal.run/analyze/lab_report` |
| `DERMATOLOGY_SERVICE_URL` | `https://your-derm.railway.app/analyze/dermatology` (CPU service; not Modal by default) |
| `ORAL_CANCER_SERVICE_URL` | **Production:** `https://….modal.run/analyze/oral_cancer` from **`modal deploy modal_app/deploy_oral_cancer.py`** (GPU T4). Optional: CPU app or Railway. |
| `REPORT_ASSEMBLY_URL` | `https://your-report-service.railway.app` (no path suffix) |

Local Docker defaults remain if these are unset.

## Smoke test

```bash
curl -sS "https://YOUR_MODAL_URL/health"
```

## See also

[`MODAL_DEPLOYMENT_PLAN.md`](../../MODAL_DEPLOYMENT_PLAN.md) in the repo parent for architecture and cost notes.
