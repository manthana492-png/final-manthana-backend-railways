# Railway gateway — Modal service URL checklist

Set these variables on the **gateway** Railway service (full HTTPS URL including `/analyze/<modality>`).

## Standard modalities

| Variable | Example path suffix |
|----------|---------------------|
| `CT_BRAIN_SERVICE_URL` | `.../analyze/ct_brain` |
| `BRAIN_MRI_SERVICE_URL` | `.../analyze/brain_mri` |
| `CARDIAC_CT_SERVICE_URL` | `.../analyze/cardiac_ct` |
| `SPINE_NEURO_SERVICE_URL` | `.../analyze/spine_neuro` |
| `ABDOMINAL_CT_SERVICE_URL` | `.../analyze/abdominal_ct` |
| `XRAY_SERVICE_URL` | `.../analyze/xray` |
| `ULTRASOUND_SERVICE_URL` | `.../analyze/ultrasound` |
| `PATHOLOGY_SERVICE_URL` | `.../analyze/pathology` |
| `CYTOLOGY_SERVICE_URL` | `.../analyze/cytology` |
| `MAMMOGRAPHY_SERVICE_URL` | `.../analyze/mammography` |
| `LAB_REPORT_SERVICE_URL` | `.../analyze/lab_report` |
| `ORAL_CANCER_SERVICE_URL` | `.../analyze/oral_cancer` |
| `ECG_SERVICE_URL` | `.../analyze/ecg` |
| `DERMATOLOGY_SERVICE_URL` | `.../analyze/dermatology` |

## Premium VISTA-3D

| Variable | Notes |
|----------|--------|
| `CT_BRAIN_VISTA_SERVICE_URL` | Modal app `manthana-ct-brain-vista` — still uses FastAPI route **`/analyze/ct_brain`** on that host |
| `PREMIUM_CT_SERVICE_URL` | Modal app `manthana-premium-ct` — unified premium CT endpoint **`/analyze/premium_ct`** |

Example:

`CT_BRAIN_VISTA_SERVICE_URL=https://YOUR_WORKSPACE--manthana-ct-brain-vista-serve.modal.run/analyze/ct_brain`

`PREMIUM_CT_SERVICE_URL=https://YOUR_WORKSPACE--manthana-premium-ct-serve.modal.run/analyze/premium_ct`

Keep existing: `JWT_SECRET`, `SUPABASE_JWT_SECRET`, `REPORT_ASSEMBLY_URL`, `GATEWAY_CORS_ORIGINS`, `XRAY_TRIAGE_POLICY`.
