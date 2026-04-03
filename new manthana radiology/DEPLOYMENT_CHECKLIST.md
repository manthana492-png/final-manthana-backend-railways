# Manthana Radiology Suite — Deployment Checklist

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Local)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Next.js 14 App (manthana-scan)                                     │   │
│  │  • Port: 3000 (dev) or 80/443 (production)                          │   │
│  │  • API Target: http://<GPU_SERVER>:8000                           │   │
│  │  • DICOM Viewer: CornerstoneJS                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │ HTTP/WebSocket                          │
│                                    ▼                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          │ Network
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (GPU Server)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  API Gateway (Port 8000)                                            │   │
│  │  • Routes to 13+ modality services                                  │   │
│  │  • JWT Authentication                                               │   │
│  │  • Auto case embeddings (Parrotlet-e)                               │   │
│  │  • PACS Proxy (Orthanc DICOM)                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│        ┌───────────────────────────┼───────────────────────────┐           │
│        │                           │                           │           │
│        ▼                           ▼                           ▼           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  X-Ray   │  │ Brain MRI│  │Cardiac CT│  │Pathology │  │  ECG     │      │
│  │ (8001)   │  │ (8002)   │  │ (8004)   │  │ (8005)   │  │ (8013)   │      │
│  │ GPU: 16GB│  │ GPU: 16GB│  │ GPU: 8GB │  │ GPU: 16GB│  │ CPU      │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │Abdominal │  │Mammograph│  │  Spine   │  │ Cytology │  │  Oral    │      │
│  │ CT (8GB) │  │ (8012)   │  │ (8010)   │  │ (8011)   │  │(8014)CPU │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │Ultrasound│  │  Dental  │  │ Lab Report│  │   Report Assembly        │  │
│  │ (8009)   │  │ (8007)   │  │ (8015)   │  │   (8020) DeepSeek        │  │
│  │ GPU: 8GB │  │ GPU: 4GB │  │ CPU+GPU*  │  │   Unified Reports        │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────────────┘  │
│                                                                            │
│  * Lab Report: Parrotlet-v-lite-4b (8GB) for document parsing              │
│  * Case Embeddings: Parrotlet-e (2GB) for similarity search              │
│                                                                            │
│  Infrastructure: Redis (queue), ChromaDB (embeddings), Orthanc (PACS)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-Deployment Requirements

### Hardware Requirements (GPU Server)

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | 1x RTX 4090 (24GB) | 2x RTX 4090 or 1x A100 (40GB) | For concurrent heavy models |
| CPU | 8 cores | 16+ cores | Model loading is CPU-intensive |
| RAM | 32GB | 64GB+ | For pathology WSI processing |
| Storage | 500GB SSD | 2TB NVMe | Model cache + uploads |
| Network | 1Gbps | 10Gbps | For DICOM transfers |

### Software Requirements

- **OS**: Ubuntu 22.04 LTS or Debian 12
- **Docker**: 24.0+ with NVIDIA Container Toolkit
- **NVIDIA Drivers**: 535+ (CUDA 12.2+)
- **NVIDIA Container Toolkit**: Latest

---

## Deployment Steps

### Step 1: Prepare GPU Server

```bash
# 1. Install NVIDIA drivers (if not already installed)
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# 2. Install Docker
sudo apt-get update
sudo apt-get install -y docker.io

# 3. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. Test GPU access in Docker
sudo docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Step 2: Deploy Backend

```bash
# 1. Clone/navigate to backend
cd /path/to/manthana-backend

# 2. Create environment file
cp .env.example .env

# 3. EDIT .env with your values:
# - DEEPSEEK_API_KEY=sk-xxx (required for lab reports & unified reports)
# - GATEWAY_PORT=8000
# - JWT_SECRET=<generate-random-32-char-string>
# - DEVICE=cuda
# - MODEL_DIR=/models

# 4. Start services (downloads ~50GB models on first run)
docker-compose up -d

# 5. Monitor logs
docker-compose logs -f gateway

# 6. Verify health
curl http://localhost:8000/health
```

### Step 3: Configure Frontend

```bash
# 1. On local machine (Windows/Mac/Linux)
cd manthana-scan

# 2. Install dependencies (if not already)
npm install

# 3. Update environment
# Edit .env.local:
NEXT_PUBLIC_GATEWAY_URL=http://<GPU_SERVER_IP>:8000

# 4. Start dev server
npm run dev

# OR for production build:
npm run build
npm start
```

---

## Environment Variables Reference

### Critical (Must Configure)

| Variable | Backend .env | Frontend .env.local | Description |
|----------|--------------|---------------------|-------------|
| `DEEPSEEK_API_KEY` | ✅ Required | ❌ | For lab reports & unified reports |
| `JWT_SECRET` | ✅ Required | ❌ | Min 32 chars, random |
| `NEXT_PUBLIC_GATEWAY_URL` | ❌ | ✅ Required | Points to GPU server |
| `GATEWAY_PORT` | ✅ 8000 | ❌ | Gateway listens on this |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_PARROTV_PARSER` | auto | always/never/auto for lab parsing |
| `VECTOR_STORE_BACKEND` | chroma | Vector DB for embeddings |
| `HF_TOKEN` | - | HuggingFace token (if models gated) |

### PACS (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `ORTHANC_USERNAME` | manthana | DICOM server username |
| `ORTHANC_PASSWORD` | - | Set strong password |
| `HOSPITAL_PACS_HOST` | - | Connect to hospital PACS |

---

## Service Verification Commands

```bash
# 1. Check all services are running
docker-compose ps

# 2. Test gateway health
curl http://localhost:8000/health

# 3. List available modalities
curl http://localhost:8000/services

# 4. Check individual service health
curl http://localhost:8001/health  # xray
curl http://localhost:8002/health  # brain_mri
# ... etc for each port

# 5. Test model loading (will download if not cached)
docker-compose logs -f body_xray | grep -i "model\|loading\|downloading"

# 6. Verify GPU usage
watch -n 1 nvidia-smi
```

---

## Testing End-to-End

### Test 1: Basic Connectivity

```bash
# From frontend machine
curl http://<GPU_SERVER>:8000/health
# Expected: {"service": "gateway", "status": "ok"}
```

### Test 2: X-Ray Analysis (via curl)

```bash
# Upload test X-ray
curl -X POST http://<GPU_SERVER>:8000/analyze \
  -F "modality=xray" \
  -F "file=@/path/to/test_xray.jpg" \
  -F "patient_id=TEST-001"
```

### Test 3: Lab Report Analysis

```bash
# Upload PDF lab report
curl -X POST http://<GPU_SERVER>:8000/analyze \
  -F "modality=lab_report" \
  -F "file=@/path/to/lab_report.pdf" \
  -F "patient_id=TEST-002"
```

### Test 4: Similar Cases (Embeddings)

```bash
# After at least one analysis, search similar
curl "http://<GPU_SERVER>:8000/cases/similar?query_text=pneumonia&top_k=5"
```

### Test 5: Frontend Integration

1. Open frontend at `http://localhost:3000`
2. Select "X-Ray" modality
3. Upload test image
4. Verify analysis completes in 5-15 seconds
5. Check findings appear in Intelligence Panel

---

## Troubleshooting Guide

### Issue: Models not downloading

**Symptoms**: Service starts but analysis times out

**Solution**:
```bash
# Check disk space
df -h

# Check HF token if models are gated
# Add to .env: HF_TOKEN=hf_xxx

# Manual download test
docker-compose exec body_xray python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('wanglab/MedRAX2', cache_dir='/models')
print('Downloaded successfully')
"
```

### Issue: GPU out of memory

**Symptoms**: `CUDA out of memory` errors

**Solution**:
```bash
# Check what's using GPU
nvidia-smi

# Reduce concurrent heavy models
# Edit docker-compose.yml: comment out some services
# Or scale down:
docker-compose stop pathology  # Largest memory user
```

### Issue: Frontend can't connect to backend

**Symptoms**: Network errors in browser console

**Solution**:
```bash
# 1. Check CORS
curl -H "Origin: http://localhost:3000" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS \
     http://<GPU_SERVER>:8000/analyze

# 2. Check firewall
sudo ufw allow 8000/tcp  # Ubuntu

# 3. Verify URL in .env.local
cat manthana-scan/.env.local
```

### Issue: Lab report parsing fails

**Symptoms**: Lab uploads return "parser not available"

**Solution**:
```bash
# Check if Parrotlet-v is loading
docker-compose logs -f lab_report | grep -i parrotlet

# Verify model registry has entry
docker-compose exec gateway python -c "
from model_registry import ModelRegistry
r = ModelRegistry()
print(r.get_active_model_id('parrotlet-v-lite-4b'))
"
```

---

## Post-Deployment Checklist

- [ ] Gateway health check passes: `curl http://<GPU>:8000/health`
- [ ] All 13+ modalities show "online" in `/services`
- [ ] Test analysis for each modality type you plan to use
- [ ] Lab report with PDF works (Parrotlet-v loads)
- [ ] Case embedding generates after analysis (check logs)
- [ ] PACS connection configured (if using)
- [ ] Frontend successfully analyzes from local machine
- [ ] Heatmaps display correctly
- [ ] Unified reports generate for multi-model
- [ ] JWT authentication working (if enabled)

---

## Production Hardening (Optional)

### Enable HTTPS
```nginx
# Add reverse proxy with SSL
server {
    listen 443 ssl;
    server_name api.manthana.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

### Resource Limits
```yaml
# Add to docker-compose.yml services
deploy:
  resources:
    limits:
      memory: 20G
    reservations:
      memory: 4G
```

### Monitoring
```bash
# Install prometheus/node-exporter for GPU monitoring
# Track: GPU utilization, memory, queue depth, API latency
```

---

## Quick Reference: Ports

| Service | Port | GPU | Purpose |
|---------|------|-----|---------|
| Gateway | 8000 | - | API entry point |
| X-Ray | 8001 | 16GB | CXR analysis |
| Brain MRI | 8002 | 16GB | Neuro imaging |
| Cardiac CT | 8004 | 8GB | Heart CT |
| Pathology | 8005 | 16GB | WSI analysis |
| Dental | 8007 | 4GB | OPG/X-ray |
| Abdominal CT | 8008 | 8GB | AAQ/BMD |
| Ultrasound | 8009 | 8GB | USG analysis |
| Spine | 8010 | 16GB | Neuro/Spine MRI |
| Cytology | 8011 | 16GB | Pap/FNA |
| Mammography | 8012 | 8GB | Breast imaging |
| ECG | 8013 | CPU | ECG analysis |
| Oral Cancer | 8014 | CPU | Lesion screening |
| Lab Report | 8015 | 8GB* | Document parsing |
| Report Asm | 8020 | - | Unified reports |
| Queue API | 8021 | - | Job queue |
| PACS Bridge | 8030 | - | DICOM proxy |
| Orthanc | 8042 | - | DICOM server |

*Parrotlet-v-lite-4b uses GPU for document parsing

---

## Support

For deployment issues:
1. Check logs: `docker-compose logs <service>`
2. Verify GPU: `nvidia-smi` inside container
3. Test model loading manually
4. Check network connectivity between frontend and GPU server
