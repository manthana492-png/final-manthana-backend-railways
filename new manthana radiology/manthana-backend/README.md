# Manthana Radiology Suite — Backend

> **India's Complete AI Radiology Second-Opinion Suite**

AI-powered decision support for radiologists, general practitioners, and rural doctors.  
12 AI services covering CT, MRI, X-Ray, Ultrasound, ECG, Mammography, Pathology, and more.

## Quick Start

```bash
# 1. Clone
git clone <your-repo-url>
cd manthana-backend

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Run (GPU server)
docker-compose up --build -d

# 4. First request triggers model downloads (~145GB, one-time)
curl -X POST http://localhost:8000/analyze \
  -H "Authorization: Bearer <jwt_token>" \
  -F "modality=ecg" \
  -F "file=@ecg_photo.jpg"
```

## Architecture

```
Gateway (:8000) → Routes by modality → 12 AI Services
                                      → Report Assembly (:8020)
                                      → Redis Queue (:6379)
```

## Services

| Service | Port | Modality |
|---------|------|----------|
| Body X-Ray | 8001 | Any X-ray (auto-detect body region) |
| Brain MRI | 8002 | Brain MRI scans |
| Cardiac CT | 8004 | Cardiac CT scans |
| Pathology | 8005 | Whole slide images |
| Abdominal CT | 8008 | Abdominal/whole-body CT |
| Ultrasound | 8009 | All ultrasound types |
| Spine/Neuro | 8010 | Spine X-ray and MRI |
| Cytology | 8011 | Cytology slides |
| Mammography | 8012 | Mammogram images |
| ECG | 8013 | ECG (photo or signal data) |
| Oral Cancer | 8014 | Oral cavity photos |

## NVIDIA NIM (Premium 3D CT + optional chat roles)

Secrets stay on the **gateway and GPU services** (Railway, Modal, Docker)—never in the Next.js client.

| Variable | Purpose |
|----------|---------|
| `NVIDIA_NIM_API_KEY` | Bearer token for hosted NIM (build.nvidia.com / nvapi key) |
| `NVIDIA_NIM_VISTA_INFER_URL` | POST URL for VISTA-3D inference (default in `.env.example`) |
| `VISTA_BACKEND` | `nim` (default) or `local` (MONAI + local checkpoint only) |
| `NIM_VISTA_IMAGE_URL` | Full HTTPS URL to a `.nii.gz` volume for tests |
| `NIM_VISTA_PUBLIC_BASE_URL` + `NIM_VISTA_PUBLIC_SYNC_DIR` | Copy exported NIfTI to a web-served directory and build the public `image` URL for NIM |

VISTA NIM expects JSON `{"image": "<https url to nifti>"}` per [NVIDIA VISTA-3D API](https://docs.nvidia.com/nim/medical/vista3d/latest/api-reference.html).  
Optional **chat** roles use `provider: nim` in `config/cloud_inference.yaml` with `nim_base_url` and `NVIDIA_NIM_API_KEY` (OpenAI-compatible `/v1/chat/completions` on `integrate.api.nvidia.com`).

## License

Proprietary — Manthana Radiology Suite  
All underlying AI models are commercially licensed (Apache 2.0 / MIT).
