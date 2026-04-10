# Manthana — NVIDIA & medical-AI compliance reference

Internal engineering document to support **DPDP Act 2023**, **CDSCO SaMD** (draft guidance, 2025), and alignment with **IEC 62304 / ISO 13485** workflows. Not legal advice; validate with your regulatory counsel.

---

## 1. NVIDIA & related component inventory

| Component | Version / pin | License | Role in Manthana |
|-----------|----------------|---------|------------------|
| NVIDIA CUDA Runtime (base image) | 12.4.0 | NVIDIA EULA | GPU inference base for Modal GPU services |
| MONAI | 1.3.0 (pinned in `modal_app/common.py`) | Apache 2.0 | CT/MRI preprocessing transforms; VISTA-3D premium path metadata |
| TorchXRayVision | service `requirements.txt` | Apache 2.0 | Chest X-ray DenseNet models (`01_body_xray`) |
| NVIDIA VISTA-3D (optional checkpoint) | User-supplied / HF bootstrap | NVIDIA AI Foundation Community License (verify per artifact) | Premium CT segmentation tier (`ct_brain_vista` Modal app) |
| nnUNet v2 | v2.4.2 (git pin in `modal_app/common.py`) | Apache 2.0 | Cardiac / spine / abdominal CT pipelines |
| TotalSegmentator | ≥2.4.0 | Apache 2.0 | Whole-body CT/MRI organ segmentation |

---

## 2. MONAI version pinning and audit trail

- **Pin:** `monai==1.3.0` via `with_monai_transforms()` in [`modal_app/common.py`](modal_app/common.py).
- **Release notes:** [MONAI 1.3.0](https://github.com/Project-MONAI/MONAI/releases/tag/1.3.0).
- **Runtime opt-in:** `MANTHANA_USE_MONAI_CT_LOADER=1` enables MONAI `LoadImage` in [`shared/preprocessing/ct_loader.py`](shared/preprocessing/ct_loader.py); film-photo directories stay on the legacy loader.
- **Volume artifacts:** MONAI bundles under `/models/monai_bundles/` (see `bootstrap_monai_bundles` in [`modal_app/bootstrap_weights.py`](modal_app/bootstrap_weights.py)).

---

## 3. IEC 62304 / ISO 13485 references (ecosystem)

- **NVIDIA Clara / Holoscan:** [NVIDIA Clara Holoscan](https://www.nvidia.com/en-us/clara/holoscan/) — platform documentation for regulated medical-device style deployments.
- **MONAI Deploy:** [MONAI Deploy App SDK](https://docs.monai.io/projects/monai-deploy-app-sdk/) — packaging and deployment patterns referenced in regulated imaging AI programs.
- **Practice:** Maintain version pins, change logs, and release artifacts for each Modal image build and gateway deploy.

---

## 4. DPDP Act 2023 — algorithmic transparency mapping (illustrative)

| Theme | Manthana implementation | NVIDIA / open-model angle |
|-------|------------------------|----------------------------|
| Transparency of processing | Structured outputs: `findings`, `pathology_scores`, `models_used`, disclaimers | Published VISTA-3D / MONAI literature and open weights (where applicable) support technical disclosure |
| Data quality / bias | Per-modality documentation; premium VISTA path documents volumetric-only input | Foundation models disclose training scale (e.g. VISTA-3D literature) |
| Purpose limitation | Gateway auth + modality routing; no training on customer uploads in default inference path | N/A |
| Security of processing | TLS in transit (Vercel ↔ Railway ↔ Modal); secrets in Modal/Railway | Standard CUDA/MONAI security advisories |
| Data minimization | Ephemeral upload dirs on gateway; Modal scale-to-zero workers | Stateless inference pattern |

---

## 5. CDSCO SaMD notes (illustrative)

- **Draft context:** CDSCO draft guidance on medical device software (2025) — confirm final rule with counsel.
- **Typical framing:** Manthana is **clinical decision support**, not a standalone diagnostic device; exact class depends on intended use and claims.
- **Standards often cited:** ISO 13485 (QMS), IEC 62304 (software lifecycle), ISO 14971 (risk management).
- **Substantial equivalence / documentation:** Use of widely cited open components (MONAI, published checkpoints) can support technical files when aligned to your labeling.

---

## 6. VISTA-3D validation references

- **Paper:** VISTA3D: Versatile Imaging SegmenTation and Annotation model for 3D CT — arXiv:2406.05285.
- **NIM docs:** [NVIDIA NIM for VISTA-3D](https://docs.nvidia.com/nim/medical/vista3d/).
- **Product wiring:** Premium gateway modality `ct_brain_vista` → `CT_BRAIN_VISTA_SERVICE_URL` → Modal app `manthana-ct-brain-vista`; service code in [`services/11_ct_brain/vista3d_integration.py`](services/11_ct_brain/vista3d_integration.py).

---

## 7. NV-Segment-CT (future optional integration)

- **Scope:** 132-class CT segmentation including tumor classes; commercial-friendly license (verify before ship).
- **Hub:** `nvidia/NV-Segment-CT` on Hugging Face.
- **Note:** Not yet wired into Manthana services; track as a separate design if product scope expands.

---

## 8. Operator commands (Modal volume)

See [`docs/MODAL_PHASE1_OPERATOR_SETUP.md`](docs/MODAL_PHASE1_OPERATOR_SETUP.md) and [`modal_app/MODAL_DEPLOYMENT_FINAL.md`](modal_app/MODAL_DEPLOYMENT_FINAL.md).

---

*Document version: aligned with manthana-backend Modal + gateway implementation.*
