# Smoke test — Modal + Railway + Vercel

Operator checklist after deploy. Code and routing are in-repo; this doc is the runbook.

## 1. Modal `/health` (cold start 30–120s)

For each deployed app, from Modal dashboard or printed URL:

```bash
curl -sS "https://<workspace>--<app-name>-serve.modal.run/health"
```

Expect `{"status":"ok"}` (or equivalent). Include **`manthana-ct-brain-vista`** after premium deploy.

## 2. Railway gateway

```bash
curl -sS "https://<gateway-host>/health"
```

## 3. Railway env wiring

Confirm every `*_SERVICE_URL` matches [`docs/RAILWAY_MODAL_SERVICE_URLS.md`](RAILWAY_MODAL_SERVICE_URLS.md).  
**Important:** `CT_BRAIN_VISTA_SERVICE_URL` must end with **`/analyze/ct_brain`** (not `ct_brain_vista` in the path).

## 4. End-to-end (Vercel)

As a logged-in user:

| Check | Action |
|-------|--------|
| Standard | X-ray photo, CT brain DICOM, ECG photo, lab PDF, oral photo |
| Premium | Pro tier → modality **CT Brain VISTA-3D** → DICOM ZIP or NIfTI only |
| Gate | Free tier → same request → expect **403** from gateway |
| Accept types | VISTA modality should not offer JPG/PNG in file picker |

## 5. After MONAI image changes

Redeploy CT/MRI Modal apps that use `with_monai_transforms` so images pick up `monai==1.3.0`:

- `deploy_ct_brain.py`, `deploy_brain_mri.py`, `deploy_cardiac_ct.py`, `deploy_spine_neuro.py`, `deploy_abdominal_ct.py`

## 6. Compliance artifact

Review [`NVIDIA_COMPLIANCE_REFERENCE.md`](../NVIDIA_COMPLIANCE_REFERENCE.md) before regulatory filings.
