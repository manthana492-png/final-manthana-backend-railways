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

## License

Proprietary — Manthana Radiology Suite  
All underlying AI models are commercially licensed (Apache 2.0 / MIT).
