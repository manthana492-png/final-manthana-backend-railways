# Manthana — Modal GPU production deployment plan (CT/MRI + X-ray)

Unified reference for **Vercel (frontend) → Railway (gateway, report assembly, Redis, optional RQ worker) → Modal (GPU + CPU inference + persistent volume)**. **Heavy GPU** (CT/MRI, X-ray, USG, pathology, oral cancer **production default**, etc.) and **bursty CPU** (ECG, dermatology, optional **oral cancer CPU** alternate) run on **Modal** with **scale-to-zero** and **$0 idle**; **always-on** pieces stay on **Railway** (gateway, report assembly, **Redis** plugin). Same Railway project/region as the gateway is recommended.

CT/MRI paths match what is implemented in [`manthana-backend/modal_app/`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app); X-ray is specified in **§15** (optional Modal app; GPU guidance below).

## Production Architecture: Vercel + Railway + Modal

```
┌─────────────────────────────────────────────────────────────────┐
│                        USERS (Browser)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼──────────────────────────────────────┐
│                    VERCEL (Frontend)                             │
│  Next.js — Manthana Oracle + Labs (manthana.quaasx108.com)        │
│  Cost: $0 (Hobby) or $20/mo (Pro)                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS REST
┌──────────────────────────▼──────────────────────────────────────┐
│                  RAILWAY (Gateway + Middleware)                  │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │  Gateway     │  │ Report       │  │ Queue API +        │     │
│  │  (FastAPI)   │  │ Assembly     │  │ Worker             │     │
│  │  Port 8000   │  │ Port 8020    │  │ Port 8021          │     │
│  └──────┬───────┘  └──────────────┘  └────────────────────┘     │
│         │                                                       │
│  ┌──────┘  Redis — Railway Redis plugin (shared REDIS_URL)        │
│  │         RQ: manthana-backend/shared/queue_client.py            │
│  │         Cost: ~$5/mo plugin + negligible CPU (see Executive E) │
└──┼──────────────────────────────────────────────────────────────┘
                           │ HTTPS (Modal web_endpoint or .remote() calls)
   │
┌──▼──────────────────────────────────────────────────────────────┐
│              MODAL (GPU + CPU Inference — Scale-to-Zero)         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Modal Volume: "manthana-model-weights"        │    │
│  │           (Persistent disk — downloaded once)            │    │
│  │                                                         │    │
│  │  Volume mounted at /models (same as MODEL_DIR in repo)  │    │
│  │  totalseg, ct_brain, synthseg, comp2comp, sybil, etc.   │    │
│  │  Total: ~7-8 GB on disk                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─ GPU Containers (scale 0→N on demand) ──────────────────┐   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ CT Brain     │  │ Brain MRI    │  │ Cardiac CT   │  │   │
│  │  │ (T4/L4)      │  │ (A10G/L4)    │  │ (T4/L4)      │  │   │
│  │  │ ~8GB VRAM    │  │ ~12GB VRAM   │  │ ~8GB VRAM    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐                     │   │
│  │  │ Spine Neuro  │  │ Abdominal CT │                     │   │
│  │  │ (T4/L4)      │  │ (A10G/L4)    │                     │   │
│  │  │ ~8GB VRAM    │  │ ~16GB VRAM   │                     │   │
│  │  └──────────────┘  └──────────────┘                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─ CPU Containers (NEW — scale 0→N, $0 idle) ────────────┐   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ ECG          │  │ Dermatology  │  │ Oral (CPU    │  │   │
│  │  │ (2 cores)    │  │ (2 cores)    │  │  alt. only)  │  │   │
│  │  │ ~4GB RAM     │  │ ~4GB RAM     │  │ ~4GB RAM     │  │   │
│  │  │ $0 when idle │  │ $0 when idle │  │ $0 when idle │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                                                         │   │
│  │  Cold start: ~3-5s (vs 8-15s GPU)                      │   │
│  │  Cost: ~$0.0000131/core/sec (~$0.47/vCPU-hr)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Idle → containers scale to 0 → $0.00/hr                       │
│  Request arrives → cold start → process → stay warm            │
│  Queue drains → scaledown_window (60s) → back to 0              │
│                                                                 │
│  Cost: $0 when idle. Pay only for compute seconds used.        │
└─────────────────────────────────────────────────────────────────┘
```

### Three-tier stack — who does what

| Tier | Platform | Responsibility |
|------|----------|----------------|
| **Frontend** | **Vercel** | Next.js app (**Oracle + Labs**); browser uses same-origin **`/api/oracle-backend`** proxy; server uses **`ORACLE_INTERNAL_URL`** to reach Oracle/gateway. See **§11 (Vercel — Manthana Next.js frontend)**. |
| **Middle / orchestration** | **Railway** | **Gateway** (JWT / Supabase token verify, `POST /analyze`, multipart forward to **Modal (GPU + CPU)** backends per `router.py`, CORS via `GATEWAY_CORS_ORIGINS`), **report_assembly** (`REPORT_ASSEMBLY_URL`), **queue API + RQ worker** when `USE_REDIS_QUEUE=1`, **Redis** via Railway Redis plugin — **always-on** middleware at low fixed cost. |
| **Inference (GPU + CPU)** | **Modal** | **GPU**: CT/MRI, X-ray, USG, pathology, mammography, lab report, **oral cancer (production)**, etc. **CPU**: ECG, dermatology, optional oral **CPU** deploy for cost. **`modal.asgi_app()`** wrapping existing FastAPI. **Modal Volume** at **`/models`** for persistent weights. **Zero idle cost for all inference.** |

**Disk vs image:** Heavy artifacts (TotalSegmentator, Comp2Comp, custom TorchScripts) → **Volume `/models`**. Small TXRV DenseNet weights (~tens of MB) → typically **baked into the Docker/Modal image** at build time (same as [`services/01_body_xray/Dockerfile`](this_studio/new%20manthana%20radiology/manthana-backend/services/01_body_xray/Dockerfile) `RUN python -c "…DenseNet…"`). Exact on-disk cache layout for `torchxrayvision` is version-dependent — verify after install if you move caches to a Volume.

---

## Codebase alignment — CT/MRI already implemented

These paths exist in the repo (deploy from `manthana-backend/`):

| Area | Location |
|------|----------|
| Modal apps (GPU) | CT/MRI: `deploy_ct_brain.py`, `deploy_brain_mri.py`, `deploy_cardiac_ct.py`, `deploy_spine_neuro.py`, `deploy_abdominal_ct.py`; also [`deploy_body_xray.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_body_xray.py), [`deploy_ultrasound.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_ultrasound.py), [`deploy_pathology.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_pathology.py), [`deploy_cytology.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_cytology.py), [`deploy_mammography.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_mammography.py), [`deploy_lab_report.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_lab_report.py), [`deploy_oral_cancer.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_oral_cancer.py) |
| Modal apps (CPU) | **ECG**: `deploy_ecg.py` → `manthana-ecg`; **Dermatology**: `deploy_dermatology.py` → `manthana-dermatology`; **Oral (optional CPU)**: `deploy_oral_cancer_cpu.py` → `manthana-oral-cancer-cpu` — use only if not using production GPU oral app. All use [`cpu_function_kwargs()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py), scale to zero, $0 idle cost. |
| Shared Modal image helpers | [`modal_app/common.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) (`studio_root()`, `with_manthana_llm_stack`, volume name, CUDA base) |
| Volume bootstrap (TotalSeg tasks) | [`modal_app/bootstrap_weights.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/bootstrap_weights.py) |
| Local Modal CLI deps | [`modal_requirements.txt`](this_studio/new%20manthana%20radiology/manthana-backend/modal_requirements.txt) |
| Operator notes | [`modal_app/README.md`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/README.md) |
| Gateway → backend URLs | [`gateway/router.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py) — `*_SERVICE_URL` for CT/MRI, X-ray, ultrasound, ECG, **pathology**, **cytology**, **mammography**, **lab_report**, **dermatology**, **oral_cancer**, etc. (Docker internal URLs if unset) |
| Gateway → report + resilience | [`gateway/main.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/main.py) — `REPORT_ASSEMBLY_URL`, retry loop on **502/503** for GPU forwards |
| LLM config on `/app` layout | [`shared/llm_router.py`](this_studio/new%20manthana%20radiology/manthana-backend/shared/llm_router.py) — walks to `packages/manthana-inference` or `MANTHANA_LLM_REPO_ROOT` |
| X-ray triage on CPU gateway | [`gateway/triage.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/triage.py) — **`txrv_utils` is lazy-imported inside `_triage_xray`** so Railway does not load PyTorch at import time when `XRAY_TRIAGE_POLICY=always_deep` (default). |

---

## Executive decisions — what to implement first (recommended)

These choices align the plan with a **bootstrapped startup**: pay only for GPU seconds, ship production CT/MRI with the existing frontend unchanged.

### A. Modal topology: **five separate Modal apps** (recommended)

Deploy **one Modal app per service**, each serving the **existing FastAPI** app from the repo (same routes as today: e.g. `/analyze/brain_mri`, `/health`). Prefer Modal’s **`modal.asgi_app()`** pattern so you do not rewrite `run_pipeline` / `run_brain_mri_pipeline` call chains.

| Approach | When it is best |
|----------|-----------------|
| **Five apps (recommended)** | Fastest path to prod, **smaller images per service** (faster pulls / often shorter cold start than one mega-image), **right-size GPU** (T4 for CT brain / cardiac / spine vs L4 for brain MRI / abdominal), **failure isolation** (e.g. Comp2Comp issues do not take down brain MRI). Matches how [manthana-backend/gateway/router.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py) already maps one URL per modality. |
| **Two consolidated GPU workers** | Only if you want **fewer deploy targets** to operate manually; expect **heavier combined images**, longer `@modal.enter()` if you preload everything, and **shared fate** on crashes. Does **not** materially reduce idle cost (both patterns scale to zero). Revisit if Modal seat/ops overhead becomes the bottleneck. |

**Verdict:** Go with **five apps** for launch; treat two-container consolidation as a **Phase 2 optimization**, not the default.

### B. Gateway → GPU: **keep multipart upload** (recommended for v1)

The gateway already forwards **`multipart/form-data`** (`file` + form fields) to each service ([manthana-backend/gateway/main.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/main.py)). Point [manthana-backend/gateway/router.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py) at **full HTTPS Modal URLs** ending in the same path (e.g. `…/analyze/ct_brain`). No base64 refactor required for launch.

| Approach | When it is best |
|----------|-----------------|
| **Multipart via gateway (recommended v1)** | Zero frontend change; works for typical DICOM/ZIP sizes you already support; **600s timeout** already set on gateway forward. |
| **S3 / GCS presign (plan section 4 Option B)** | Add when you routinely exceed practical limits or see timeouts on huge studies; not required day one. |

### C. Railway CPU side: **gateway + report_assembly** (implemented)

**ECG** and **dermatology** are **not** required on Railway if you deploy them on **Modal CPU** (**§17–§22**). **Oral cancer** production default is **Modal GPU** (**§23**); optional **Modal CPU** oral app for cost — the gateway still sets a **single** `ORAL_CANCER_SERVICE_URL`.

The gateway uses **`REPORT_ASSEMBLY_URL`** (default `http://report_assembly:8020`) for `/assemble_report` and `/assemble_unified_report` — see [`gateway/main.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/main.py). On Railway, set `REPORT_ASSEMBLY_URL` to your **report_assembly** service origin (no trailing slash).

### D. Volume vs download-on-boot

**One Modal Volume** (e.g. `manthana-model-weights`) populated by a **one-off download job** + `modal volume put` for proprietary TorchScripts. Set **`MODEL_DIR=/models`** (mount the volume there) so [manthana-backend/shared/totalseg_runner.py](this_studio/new%20manthana%20radiology/manthana-backend/shared/totalseg_runner.py) and your env vars resolve to the mounted path — **no re-download on each container start**.

### E. Redis on Railway (production default for Manthana Labs)

**Decision:** Use **Railway’s Redis** (database plugin) for the optional **RQ** job queue. Keeps latency low vs off-platform Redis, avoids running Redis on your own CPU (e.g. Kamatera), and matches the rest of the processing layer on Railway.

| Item | Detail |
|------|--------|
| **Code** | [`shared/queue_client.py`](this_studio/new%20manthana%20radiology/manthana-backend/shared/queue_client.py) — `redis.from_url(REDIS_URL)` + RQ `Queue`. Worker: [`services/queue/worker.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/queue/worker.py). |
| **Enable** | `USE_REDIS_QUEUE=1` on every service that **enqueues** or **consumes** jobs (gateway and/or `queue` API + **worker** service). |
| **URL** | Railway dashboard → **New** → **Database** → **Add Redis** (or **Redis** from templates). Copy **`REDIS_URL`** (often `redis://…` or `rediss://…` with TLS). Paste **identical** value into: **gateway**, **queue_api** (if separate), **queue_worker**. |
| **TLS** | If Railway gives `rediss://`, `redis-py` handles TLS from the URL; no code change. |
| **Disable** | Omit `USE_REDIS_QUEUE` or set `0` — gateway runs **without** Redis (sync `/analyze` only); no Redis cost. |
| **Health** | Queue status: [`services/queue/main.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/queue/main.py) (deploy as its own Railway service if you expose queue HTTP). |

**Production checklist (Redis):**

1. Redis plugin added to the **same Railway project** as gateway (recommended).
2. `REDIS_URL` set on **all** consumers (worker + any service calling `submit_job`).
3. Worker service runs continuously (`services/queue` Dockerfile / start command from repo).
4. After deploy: submit a test job or hit queue health; confirm no connection errors in logs.

---

## 1. Model Weight Inventory (CT/MRI Only)

| # | Model | Used By | Format | Est. Size | GPU Needed |
|---|-------|---------|--------|-----------|------------|
| 1 | **TotalSegmentator** (nnU-Net) — tasks: `total`, `total_mr`, `heartchambers`, `vertebrae_body`, `vertebrae_mr` | Brain MRI, Cardiac CT, Abdominal CT, Spine Neuro | nnU-Net checkpoints (auto-download) | ~3.5 GB (all tasks combined) | 4-8 GB VRAM |
| 2 | **CT Brain ICH** main classifier | CT Brain | TorchScript `.pt` | ~300-500 MB | 2-4 GB VRAM |
| 3 | **CT Brain ICH subtype** classifier | CT Brain | TorchScript `.pt` | ~100-200 MB | 1-2 GB VRAM |
| 4 | **CT Brain volumetric segmentation** (nnU-Net-style) | CT Brain | TorchScript `.pt` | ~200-400 MB | 2-4 GB VRAM |
| 5 | **SynthSeg** (brain parcellation) | Brain MRI | TensorFlow/Keras `.h5` | ~350-500 MB | 2-4 GB VRAM |
| 6 | **WMH segmentation** model | Brain MRI | TorchScript `.pt` | ~100-200 MB | CPU (runs on CPU) |
| 7 | **Brain lesion segmentation** (BraTS-style) | Brain MRI | TorchScript/ONNX `.pt`/`.onnx` | ~200-400 MB | CPU (loaded on CPU) |
| 8 | **Comp2Comp** (body composition) — pipelines: `spine`, `liver_spleen_pancreas`, `spine_muscle_adipose_tissue` | Abdominal CT | CLI + internal checkpoints | ~1-2 GB | 4-8 GB VRAM |
| 9 | **Sybil** (lung cancer risk) | Abdominal CT (chest region) | PyTorch ensemble | ~400-600 MB | 2-4 GB VRAM |
| **TOTAL** | | | | **~7-8 GB** | |

---

## 2. Modal Volume — One-Time Download, Persistent Storage

### How it works

```
First deploy (one-time):
  modal run download_weights.py
    → Downloads ALL model weights into Modal Volume
    → Stored on Modal's distributed filesystem
    → Persists across container restarts, deploys, code updates
    → NEVER re-downloaded unless you explicitly run it again

Every inference request:
  Container starts → Volume already mounted at /weights
    → Models loaded from /weights into GPU VRAM
    → Process request → return result
    → If no more requests → container shuts down
    → Weights STAY on Volume (disk), only GPU VRAM freed
```

### Volume structure on Modal

```
/weights/
├── totalseg/
│   ├── nnUNet_results/          # nnU-Net task checkpoints
│   │   ├── Dataset291_TotalSegmentator_total/
│   │   ├── Dataset292_TotalSegmentator_total_mr/
│   │   ├── Dataset293_TotalSegmentator_heartchambers/
│   │   └── Dataset294_TotalSegmentator_vertebrae/
│   └── .totalsegmentator/       # config cache
├── ct_brain/
│   ├── ich_main.pt              # main ICH TorchScript
│   ├── ich_subtype.pt           # subtype classifier
│   └── segmentation.pt          # volumetric seg
├── synthseg/
│   └── SynthSeg/                # cloned repo + weights
├── comp2comp/
│   └── Comp2Comp/               # cloned repo + internal models
├── sybil/
│   └── sybil_ensemble/          # model cache
├── wmh/
│   └── wmh_model.pt
└── brain_lesion/
    └── lesion_model.pt
```

### Download script (run once)

```python
# modal_app/download_weights.py
import modal

volume = modal.Volume.from_name("manthana-model-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04")
    .apt_install("python3", "python3-pip", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.0",
        "TotalSegmentator>=2.4.0",
        "sybil>=1.6.0",
        "nnunetv2",
        "numpy", "nibabel",
    )
)

app = modal.App("manthana-download-weights")

WEIGHTS_DIR = "/weights"

@app.function(
    image=image,
    volumes={WEIGHTS_DIR: volume},
    timeout=3600,  # allow up to 1 hour for downloads
    gpu="T4",      # some models need GPU context to download
)
def download_all():
    import subprocess, os

    # 1. TotalSegmentator — force download all task weights
    os.environ["TOTALSEG_WEIGHTS_PATH"] = f"{WEIGHTS_DIR}/totalseg"
    from totalsegmentator.python_api import totalsegmentator
    # Trigger weight downloads by running a tiny dummy (or use their download API)
    subprocess.run([
        "TotalSegmentator", "--download_all_tasks"
    ], check=False)

    # 2. Sybil ensemble
    os.environ["SYBIL_CACHE"] = f"{WEIGHTS_DIR}/sybil"
    from sybil import Sybil
    _ = Sybil("sybil_ensemble")  # triggers download

    # 3. Comp2Comp — git clone
    comp2comp_dir = f"{WEIGHTS_DIR}/comp2comp/Comp2Comp"
    if not os.path.exists(comp2comp_dir):
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/StanfordMIMI/Comp2Comp",
            comp2comp_dir
        ], check=True)
        subprocess.run(["pip", "install", "-e", comp2comp_dir], check=True)

    # 4. SynthSeg
    synthseg_dir = f"{WEIGHTS_DIR}/synthseg/SynthSeg"
    if not os.path.exists(synthseg_dir):
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/BBillot/SynthSeg",
            synthseg_dir
        ], check=True)

    # 5. CT Brain models — these are YOUR custom weights,
    #    upload them manually:
    #    modal volume put manthana-model-weights local/ich_main.pt ct_brain/ich_main.pt
    #    modal volume put manthana-model-weights local/ich_subtype.pt ct_brain/ich_subtype.pt
    #    modal volume put manthana-model-weights local/segmentation.pt ct_brain/segmentation.pt

    # 6. WMH + Brain Lesion models — same, upload manually:
    #    modal volume put manthana-model-weights local/wmh_model.pt wmh/wmh_model.pt
    #    modal volume put manthana-model-weights local/lesion_model.pt brain_lesion/lesion_model.pt

    volume.commit()
    print("All weights downloaded and committed to volume.")
```

**Upload custom weights (CT Brain, WMH, Lesion):**
```bash
# One-time upload of your own trained models
modal volume put manthana-model-weights ./models/ich_main.pt ct_brain/ich_main.pt
modal volume put manthana-model-weights ./models/ich_subtype.pt ct_brain/ich_subtype.pt
modal volume put manthana-model-weights ./models/segmentation.pt ct_brain/segmentation.pt
modal volume put manthana-model-weights ./models/wmh_model.pt wmh/wmh_model.pt
modal volume put manthana-model-weights ./models/lesion_model.pt brain_lesion/lesion_model.pt
```

---

## 3. Modal Service Architecture — Per-Service Classes

### Design Principles

1. **One `modal.Cls` per service** — each CT/MRI pipeline is its own class
2. **`@modal.enter()`** loads models into VRAM once per container startup
3. **`scaledown_window=60`** — container stays warm 60s after last request (handles back-to-back scans)
4. **`min_containers=0`** — zero cost when idle
5. **`max_containers=3`** — cap concurrent GPUs (cost control)
6. **`web_endpoint`** — HTTP endpoint that Railway gateway calls directly

### Container Lifecycle

```
No requests → 0 containers ($0.00)
    │
    ▼ Request arrives
Cold start (~8-15s):
    1. Container boots (~1s)
    2. @modal.enter() runs:
       - Mounts Volume at /weights
       - Loads model weights from disk → GPU VRAM (~5-12s)
    3. Ready to serve
    │
    ▼ Process request (~5-30s depending on scan)
    │
    ▼ Return result
    │
    ▼ Wait for next request (scaledown_window=60s)
    │
    ├─ Another request within 60s? → Process immediately (warm, no cold start)
    │
    └─ No request for 60s → Container shuts down → 0 containers ($0.00)
        (Weights stay on Volume disk, only GPU VRAM freed)
```

### Example: CT Brain Service on Modal

```python
# modal_app/ct_brain_service.py
import modal

volume = modal.Volume.from_name("manthana-model-weights")
WEIGHTS = "/weights"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04")
    .apt_install("python3", "python3-pip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.0", "numpy>=1.24", "scipy>=1.12",
        "pydicom>=2.4", "opencv-python-headless>=4.8",
        "nibabel>=5.0", "fastapi", "uvicorn",
    )
    .copy_local_dir("services/11_ct_brain", "/app/service")
    .copy_local_dir("shared", "/app/shared")
)

app = modal.App("manthana-ct-brain")

@app.cls(
    image=image,
    gpu="T4",                   # 16GB VRAM, $0.59/hr
    volumes={WEIGHTS: volume},
    scaledown_window=60,        # stay warm 60s after last request
    min_containers=0,           # scale to zero when idle
    max_containers=3,           # cap cost
    timeout=300,                # 5-min max per request
    secrets=[modal.Secret.from_name("manthana-env")],
)
class CTBrainService:
    @modal.enter()
    def load_models(self):
        """Runs ONCE when container starts. Loads weights into GPU."""
        import sys
        sys.path.insert(0, "/app/service")
        sys.path.insert(0, "/app/shared")
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load main ICH model
        ich_path = f"{WEIGHTS}/ct_brain/ich_main.pt"
        self.ich_model = torch.jit.load(ich_path, map_location=device)
        self.ich_model.eval()

        # Load subtype model (optional)
        sub_path = f"{WEIGHTS}/ct_brain/ich_subtype.pt"
        self.subtype_model = None
        try:
            self.subtype_model = torch.jit.load(sub_path, map_location=device)
            self.subtype_model.eval()
        except Exception:
            pass

        # Load segmentation model (optional)
        seg_path = f"{WEIGHTS}/ct_brain/segmentation.pt"
        self.seg_model = None
        try:
            self.seg_model = torch.jit.load(seg_path, map_location=device)
            self.seg_model.eval()
        except Exception:
            pass

        self.device = device
        print(f"CT Brain models loaded on {device}")

    @modal.web_endpoint(method="POST")
    async def analyze(self, request: dict):
        """Called by Railway gateway. Runs full CT Brain pipeline."""
        import sys
        sys.path.insert(0, "/app/service")
        sys.path.insert(0, "/app/shared")
        from inference import run_pipeline

        # Pass pre-loaded models via environment/globals
        import os
        os.environ["CT_BRAIN_TORCHSCRIPT_PATH"] = f"{WEIGHTS}/ct_brain/ich_main.pt"
        os.environ["CT_BRAIN_SUBTYPE_MODEL_PATH"] = f"{WEIGHTS}/ct_brain/ich_subtype.pt"
        os.environ["CT_BRAIN_SEGMENTATION_MODEL_PATH"] = f"{WEIGHTS}/ct_brain/segmentation.pt"

        result = await run_pipeline(
            volume_data=request.get("volume_data"),
            clinical_notes=request.get("clinical_notes", ""),
            # ... other params
        )
        return result
```

### Example: Brain MRI Service (heavier, needs A10G/L4)

```python
# modal_app/brain_mri_service.py
import modal

volume = modal.Volume.from_name("manthana-model-weights")
WEIGHTS = "/weights"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04")
    .apt_install("python3", "python3-pip", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch>=2.0", "TotalSegmentator==2.3.0",
        "numpy>=1.24", "pydicom>=2.4", "nibabel>=5.0",
        "opencv-python-headless>=4.8", "monai",
        "fastapi", "uvicorn", "openai", "anthropic",
    )
    .run_commands("pip install git+https://github.com/MIC-DKFZ/nnUNet.git@v2.4.2")
    .copy_local_dir("services/02_brain_mri", "/app/service")
    .copy_local_dir("shared", "/app/shared")
)

app = modal.App("manthana-brain-mri")

@app.cls(
    image=image,
    gpu="L4",                   # 24GB VRAM, $0.80/hr — good for TotalSeg + SynthSeg
    volumes={WEIGHTS: volume},
    scaledown_window=60,
    min_containers=0,
    max_containers=2,
    timeout=600,                # 10-min max (MRI pipelines are heavier)
    secrets=[modal.Secret.from_name("manthana-env")],
)
class BrainMRIService:
    @modal.enter()
    def load_models(self):
        import os
        # Point TotalSegmentator to pre-downloaded weights
        os.environ["TOTALSEG_WEIGHTS_PATH"] = f"{WEIGHTS}/totalseg"
        os.environ["SYNTHSEG_SCRIPT"] = f"{WEIGHTS}/synthseg/SynthSeg/scripts/commands/SynthSeg_predict.py"
        os.environ["WMH_MODEL_PATH"] = f"{WEIGHTS}/wmh/wmh_model.pt"
        os.environ["BRAIN_LESION_MODEL_PATH"] = f"{WEIGHTS}/brain_lesion/lesion_model.pt"
        print("Brain MRI env configured, models on Volume ready")

    @modal.web_endpoint(method="POST")
    async def analyze(self, request: dict):
        import sys
        sys.path.insert(0, "/app/service")
        sys.path.insert(0, "/app/shared")
        from inference import run_brain_mri_pipeline
        return await run_brain_mri_pipeline(...)
```

---

## 4. Gateway Integration (Railway → Modal)

The Railway-hosted gateway resolves backends via **`SERVICE_MAP` in** [manthana-backend/gateway/router.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py) (not `main.py`). For production, **override each CT/MRI entry with a full URL** (scheme + host + path) so the existing `httpx` multipart forward in [manthana-backend/gateway/main.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/main.py) stays unchanged.

### Gateway routing change

```python
# gateway/router.py — pattern: env override with Docker default

import os

def _u(env_key: str, default: str) -> str:
    return os.getenv(env_key, default).strip() or default

SERVICE_MAP = {
    # ... other modalities unchanged ...
    "brain_mri": _u("BRAIN_MRI_SERVICE_URL", "http://brain_mri:8002/analyze/brain_mri"),
    "cardiac_ct": _u("CARDIAC_CT_SERVICE_URL", "http://cardiac_ct:8004/analyze/cardiac_ct"),
    "abdominal_ct": _u("ABDOMINAL_CT_SERVICE_URL", "http://abdominal_ct:8008/analyze/abdominal_ct"),
    "spine_neuro": _u("SPINE_NEURO_SERVICE_URL", "http://spine_neuro:8010/analyze/spine_neuro"),
    "ct_brain": _u("CT_BRAIN_SERVICE_URL", "http://ct_brain:8017/analyze/ct_brain"),
}

# Railway example (.env):
# CT_BRAIN_SERVICE_URL=https://<workspace>--<app>-<stub>.modal.run/analyze/ct_brain
# BRAIN_MRI_SERVICE_URL=https://.../analyze/brain_mri
# (same idea for cardiac_ct, spine_neuro, abdominal_ct)
```

**Health checks:** [manthana-backend/gateway/main.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/main.py) builds `health_url` with `url.rsplit("/analyze", 1)[0] + "/health"`. That remains valid for Modal URLs as long as the analyze URL contains `/analyze/...`.

**Report assembly:** add `REPORT_ASSEMBLY_URL` (default `http://report_assembly:8020`) and use it for `/assemble_report` and `/assemble_unified_report` instead of a fixed hostname.

### Data flow for a scan

```
1. User uploads DICOM/images → Vercel frontend
2. Frontend POST /analyze → Railway gateway (port 8000)
3. Gateway validates, preprocesses, determines modality
4. Gateway POST to Modal web endpoint (HTTPS):
   - Sends: volume data (base64), clinical notes, patient context
   - Modal cold-starts container if needed (~8-15s first time)
   - Modal loads models from Volume → GPU VRAM (~5-10s)
   - Runs inference pipeline (~5-30s)
   - Returns JSON result
5. Gateway enriches result → sends to Report Assembly
6. Gateway returns final report → Frontend
7. Modal container stays warm 60s, then scales to 0
```

### Important: File transfer strategy

DICOM files can be large (100-500MB). Two options:

**Option A: Base64 in request body (simple, for <50MB)**
```python
# Gateway sends volume data as base64
import base64
payload = {
    "volume_b64": base64.b64encode(volume_bytes).decode(),
    "clinical_notes": notes,
    "patient_context": ctx,
}
response = await httpx.post(modal_url, json=payload, timeout=300)
```

**Option B: Shared cloud storage (for large DICOMs, recommended)**
```python
# Gateway uploads to S3/GCS → passes URL to Modal
# Modal downloads from S3 (fast, within same cloud region)
import boto3
s3 = boto3.client("s3")
key = f"uploads/{job_id}/volume.npy.gz"
s3.upload_fileobj(volume_file, "manthana-uploads", key)

payload = {
    "volume_s3_key": key,
    "clinical_notes": notes,
}
response = await httpx.post(modal_url, json=payload, timeout=300)
```

---

## 5. GPU Selection Per Service

| Service | Models Loaded | Peak VRAM | Recommended GPU | Cost/hr |
|---------|---------------|-----------|-----------------|---------|
| **CT Brain** | ICH main + subtype + seg | ~6-8 GB | **T4** (16GB) | $0.59 |
| **Brain MRI** | TotalSeg(total_mr) + SynthSeg + WMH + Lesion | ~10-14 GB | **L4** (24GB) | $0.80 |
| **Cardiac CT** | TotalSeg(heartchambers) | ~4-6 GB | **T4** (16GB) | $0.59 |
| **Spine Neuro** | TotalSeg(vertebrae) | ~4-6 GB | **T4** (16GB) | $0.59 |
| **Abdominal CT** | TotalSeg(total) + Comp2Comp + Sybil | ~12-16 GB | **L4** (24GB) | $0.80 |

**Why L4 over A10G?** L4 has 24GB VRAM like A10G but costs $0.80/hr vs $1.10/hr — 27% cheaper with similar performance for inference workloads.

---

## 6. Cost Estimation

### Per-Scan GPU Cost

| Service | GPU | Time per scan | Cost per scan |
|---------|-----|---------------|---------------|
| CT Brain | T4 ($0.59/hr) | ~15-30s | **$0.003-0.005** |
| Brain MRI | L4 ($0.80/hr) | ~30-90s | **$0.007-0.020** |
| Cardiac CT | T4 ($0.59/hr) | ~20-40s | **$0.003-0.007** |
| Spine Neuro | T4 ($0.59/hr) | ~15-30s | **$0.003-0.005** |
| Abdominal CT | L4 ($0.80/hr) | ~60-120s | **$0.013-0.027** |

**Average cost per scan: ~$0.005-0.015** (depending on modality)

### Monthly Projections

| Scans/month | Avg GPU cost/scan | Modal GPU cost | Railway | Vercel | LLM (OpenRouter) | **Total** |
|-------------|-------------------|----------------|---------|--------|-------------------|-----------|
| **50** (light) | $0.01 | $0.50 | $5-10 | $0 | $2-5 | **~$8-16/mo** |
| **200** (moderate) | $0.01 | $2.00 | $10-15 | $0-20 | $10-20 | **~$22-57/mo** |
| **500** (busy) | $0.01 | $5.00 | $15-20 | $20 | $25-50 | **~$65-95/mo** |
| **2000** (high) | $0.01 | $20.00 | $20-30 | $20 | $100-200 | **~$160-270/mo** |

### Cost Breakdown Detail

**Modal (GPU inference):**
- Volume storage: **FREE** (included in Modal plan — no separate charge for stored Volumes)
- GPU compute: Per-second billing, $0 when idle
- Free tier: $30/mo credits (Starter plan) — covers ~50-100 scans free
- Team plan ($250/mo): for production scale, includes $100 credits + 50 GPU concurrency

**Railway (Gateway + Middleware + Redis plugin) — Always-On:**
- Pro plan: $20/mo (includes $20 credits) — typical for production
- Gateway service: ~0.25 vCPU + 512MB RAM when idle ≈ **$5–8/mo**
- Report assembly service: **~$3–6/mo** (light always-on for low-latency reports)
- **Redis (Railway plugin):** ~**$5/mo** (fixed add-on)
- Queue **worker** (if `USE_REDIS_QUEUE=1`): small always-on ≈ **$3–6/mo**
- **Estimated Railway stack: ~$15–25/mo** (fixed always-on cost regardless of scan volume)

**Modal (CPU inference — NEW: ECG + Dermatology + Oral Cancer):**
- Per-core-second billing: **$0.0000131/core/sec** (~$0.047/vCPU-hour)
- Scale to zero: **$0 when idle** (unlike Railway always-on)
- Example: 2-core container for 30 seconds = **$0.0008 per scan**
- Light usage (100 scans/mo): **~$0.08/mo** vs $15-24/mo on Railway
- Moderate usage (1000 scans/mo): **~$0.80/mo** vs $15-24/mo on Railway
- **Key benefit**: Bursty workloads cost dramatically less on Modal CPU vs Railway always-on

**Vercel (Frontend):**
- Hobby: $0/mo (100GB bandwidth)
- Pro: $20/mo (1TB bandwidth, custom domains, team)

**OpenRouter (LLM narratives):**
- Per-scan narrative: ~$0.005-0.01 per call
- ~$5-20/mo at moderate usage

---

## 7. Cold Start Optimization Strategies

### Strategy 1: Optimized Docker images (mandatory)

```python
# Pre-install ALL pip packages in the image build step
# This avoids pip install on every container boot
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04")
    .apt_install(...)
    .pip_install(...)  # All deps installed at IMAGE BUILD time
    .copy_local_dir(...)
)
# Image is cached by Modal — only rebuilt when definition changes
```

### Strategy 2: Lazy model loading (smart)

```python
@app.cls(...)
class CTBrainService:
    @modal.enter()
    def setup(self):
        # Load ONLY the always-needed model immediately
        self.ich_model = torch.jit.load(f"{WEIGHTS}/ct_brain/ich_main.pt")
        self.ich_model.eval()

        # Lazy-load optional models on first use
        self._subtype_model = None
        self._seg_model = None

    @property
    def subtype_model(self):
        if self._subtype_model is None:
            self._subtype_model = torch.jit.load(f"{WEIGHTS}/ct_brain/ich_subtype.pt")
            self._subtype_model.eval()
        return self._subtype_model
```

### Strategy 3: Warm buffer for peak hours (optional, costs money)

```python
# Keep 1 container warm during business hours (9am-9pm IST)
# This eliminates cold starts but costs ~$7/day per service
@app.cls(
    gpu="T4",
    min_containers=1,  # Always 1 warm container
    # OR use scheduled scaling:
)
```

**Recommendation: Do NOT keep warm containers initially.** The 8-15s cold start is acceptable for radiology workflows where the user has already uploaded files and is waiting. Only add warm containers if user complaints arise.

---

## 8. Consolidation Option — Fewer Containers, Lower Cost

Instead of 5 separate Modal classes, you can consolidate services that share the same GPU:

### Option: 2 Modal containers instead of 5

```
Container A (T4, $0.59/hr):
  - CT Brain (ICH models)
  - Cardiac CT (TotalSeg heartchambers)
  - Spine Neuro (TotalSeg vertebrae)

Container B (L4, $0.80/hr):
  - Brain MRI (TotalSeg MR + SynthSeg + WMH + Lesion)
  - Abdominal CT (TotalSeg total + Comp2Comp + Sybil)
```

**Pros:**
- Fewer cold starts (if user sends CT Brain then Cardiac CT, same container is warm)
- Simpler deployment (2 Modal apps instead of 5)
- TotalSegmentator weights loaded once, shared across pipelines

**Cons:**
- Larger container = slightly longer cold start
- One failure affects co-hosted services
- Can't scale services independently

**Recommendation:** Prefer **five separate Modal apps** for initial production (see **Executive decisions** at top). Use **two consolidated workers** only as a later ops-driven simplification — it does not reduce idle spend and can increase image size and blast radius.

---

## 9. Implementation Roadmap (high level)

See **§14 Complete implementation plan** for the full step-by-step checklist, file-level tasks, and acceptance criteria. Summary:

| Phase | Focus |
|-------|--------|
| 0–1 | Modal account, Volume, one-off weight population, secrets |
| 2 | Five Modal apps (ASGI), deploy, record HTTPS URLs |
| 3 | Gateway `router.py` + `REPORT_ASSEMBLY_URL`, Railway |
| 4 | Vercel frontend env + CORS |
| 5 | E2E tests, monitoring, cost alerts |

---

## 10. File Structure for Modal Deployment

```
manthana-backend/
├── modal_app/                          # NEW — Modal deployment code
│   ├── common.py                       # Shared: volume, image definitions
│   ├── download_weights.py             # One-time weight downloader
│   ├── ct_brain_service.py             # CT Brain Modal class
│   ├── brain_mri_service.py            # Brain MRI Modal class
│   ├── cardiac_ct_service.py           # Cardiac CT Modal class
│   ├── spine_neuro_service.py          # Spine Neuro Modal class
│   └── abdominal_ct_service.py         # Abdominal CT Modal class
├── services/                           # EXISTING — unchanged
│   ├── 02_brain_mri/
│   ├── 04_cardiac_ct/
│   ├── 08_abdominal_ct/
│   ├── 10_spine_neuro/
│   └── 11_ct_brain/
├── shared/                             # EXISTING — unchanged
│   ├── totalseg_runner.py
│   ├── synthseg_runner.py
│   ├── comp2comp_runner.py
│   ├── llm_router.py
│   └── ...
└── gateway/                            # EXISTING — modified for Modal URLs
    └── main.py
```

---

## 11. Key Environment Variables

### Modal Secrets (`manthana-env`)

```env
# LLM / Cloud AI
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_API_KEY_2=sk-or-...

# Model paths (Modal Volume mounted at /models — see §14.2)
MODEL_DIR=/models
CT_BRAIN_TORCHSCRIPT_PATH=/models/ct_brain/ich_main.pt
CT_BRAIN_SUBTYPE_MODEL_PATH=/models/ct_brain/ich_subtype.pt
CT_BRAIN_SEGMENTATION_MODEL_PATH=/models/ct_brain/segmentation.pt
WMH_MODEL_PATH=/models/wmh/wmh_model.pt
BRAIN_LESION_MODEL_PATH=/models/brain_lesion/lesion_model.pt
SYNTHSEG_SCRIPT=/models/synthseg/SynthSeg/scripts/commands/SynthSeg_predict.py
COMP2COMP_DIR=/models/Comp2Comp

# Device
CT_BRAIN_DEVICE=cuda
DEVICE=cuda
TOTALSEG_DEVICE=gpu
```

### Railway Environment

```env
# Full URL including /analyze/<modality> path (matches router.py SERVICE_MAP)
CT_BRAIN_SERVICE_URL=https://your-ws--....modal.run/analyze/ct_brain
BRAIN_MRI_SERVICE_URL=https://your-ws--....modal.run/analyze/brain_mri
CARDIAC_CT_SERVICE_URL=https://your-ws--....modal.run/analyze/cardiac_ct
SPINE_NEURO_SERVICE_URL=https://your-ws--....modal.run/analyze/spine_neuro
ABDOMINAL_CT_SERVICE_URL=https://your-ws--....modal.run/analyze/abdominal_ct

# CPU narrative assembly (Railway service or internal URL)
REPORT_ASSEMBLY_URL=https://your-report-assembly.up.railway.app
# or http://report_assembly:8020 if still on Docker network

# Optional: async RQ queue — Railway Redis plugin (same REDIS_URL on gateway + queue API + worker)
USE_REDIS_QUEUE=1
# In Railway: Redis service → Variables → copy REDIS_URL, or use "Reference" to inject into gateway/worker
REDIS_URL=redis://default:...@redis.railway.internal:6379

# LLM keys (gateway + report_assembly)
OPENROUTER_API_KEY=sk-or-...

# Auth: verify browser tokens from Vercel/Supabase (Manthana Labs frontend)
SUPABASE_JWT_SECRET=<Supabase Dashboard → Project Settings → API → JWT Secret>
# Optional: SUPABASE_JWT_ISS=https://<project-ref>.supabase.co/auth/v1
```

### Vercel — Manthana Next.js frontend (Oracle + Labs only)

**Production domain:** `https://manthana.quaasx108.com` (DNS → Vercel; assign domain in Vercel project settings).

**Product scope:** The deployed app is **Manthana Oracle** + **Manthana Labs** only. **Manthana Web** (medical search) and **Med Deep Research** are **not** connected in this production cut — keep the frontend feature flag **`NEXT_PUBLIC_MANTHANA_WEB_LOCKED=true`** and **do not** set **`NEXT_PUBLIC_FULL_MANTHANA_NAV`** (slim sidebar).

**Repository path:** [`this_studio/oracle-2/frontend-manthana/manthana`](this_studio/oracle-2/frontend-manthana/manthana) — env template: **`.env.example`**.

#### Vercel environment variables (auth + Oracle — copy in one pass)

| Variable | Required | Notes |
|----------|----------|--------|
| `NEXT_PUBLIC_SUPABASE_URL` | **Yes** | Supabase → Settings → API → Project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` *or* `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY` | **Yes** | One public key; matches local `.env.local` |
| `SUPABASE_SERVICE_ROLE_KEY` | **Yes** (prod) | **Server only** — Labs trial persistence, webhooks; never expose to client |
| `NEXT_PUBLIC_APP_URL` | **Yes** | `https://manthana.quaasx108.com` |
| `NEXT_PUBLIC_APP_DOMAIN` | Recommended | `manthana.quaasx108.com` |
| `ORACLE_INTERNAL_URL` | **Yes** | HTTPS base where Vercel **server** reaches Oracle (Railway gateway or Oracle host) — **not** `localhost` |
| `NEXT_PUBLIC_GATEWAY_URL` | Yes | `/api/oracle-backend` |
| `NEXT_PUBLIC_API_URL` | Yes | `/api/oracle-backend` |
| `NEXT_PUBLIC_ORACLE_API_URL` | Yes | `/api/oracle-backend` |
| `NEXT_PUBLIC_WEB_API_URL` | Yes | `/api/oracle-backend` (legacy var; same proxy) |
| `NEXT_PUBLIC_RESEARCH_API_URL` | Yes | `/api/oracle-backend` |
| `NEXT_PUBLIC_ANALYSIS_API_URL` | Yes | `/api/oracle-backend` |
| `NEXT_PUBLIC_CLINICAL_API_URL` | Yes | `/api/oracle-backend` |
| `NEXT_PUBLIC_MANTHANA_WEB_LOCKED` | Yes | `true` |
| Razorpay `RAZORPAY_*` + `NEXT_PUBLIC_RAZORPAY_KEY_ID` | If billing on | See `.env.example` |
| `AWS_*` | If SES / hooks | Server-side only |

After saving env vars: **Redeploy** the latest deployment (or trigger a new build).

#### Supabase Dashboard (Authentication → URL configuration)

- **Site URL:** `https://manthana.quaasx108.com`
- **Redirect URLs:** include `https://manthana.quaasx108.com/**` and explicitly `https://manthana.quaasx108.com/auth/callback`; keep `http://localhost:3001/**` for local dev.

#### Google Sign-In (optional)

- Google Cloud OAuth client → **Authorized redirect URI:** `https://<project-ref>.supabase.co/auth/v1/callback`
- Supabase → Authentication → Providers → Google: Client ID + Secret

#### Credential that lives on Railway, not Vercel

- **`SUPABASE_JWT_SECRET`** (Supabase → Settings → API → **JWT Secret**) — set on the **Railway gateway** as `SUPABASE_JWT_SECRET` so `Authorization: Bearer <supabase access token>` from the Labs upload flow validates. This is **separate** from the anon/publishable key on Vercel.

---

## 12. Production Checklist

- [ ] Modal Volume created and weights downloaded (one-time)
- [ ] Custom model weights uploaded via `modal volume put`
- [ ] Each service tested individually: `modal serve <service>.py`
- [ ] Each service deployed: `modal deploy <service>.py`
- [ ] [manthana-backend/gateway/router.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py): set `*_SERVICE_URL` for CT/MRI, **`XRAY_SERVICE_URL`**, **`ULTRASOUND_SERVICE_URL`** (Modal GPU), **`ECG_SERVICE_URL`**, **`DERMATOLOGY_SERVICE_URL`**, **`ORAL_CANCER_SERVICE_URL`** (**production:** **Modal GPU** `deploy_oral_cancer.py` → `…/analyze/oral_cancer`; optional: Modal CPU or Railway), **`PATHOLOGY_SERVICE_URL`**, **`CYTOLOGY_SERVICE_URL`**, **`MAMMOGRAPHY_SERVICE_URL`** (Modal GPU) to full HTTPS analyze URLs
- [x] `REPORT_ASSEMBLY_URL` in gateway for `/report` flows (code in `main.py`; set env on Railway)
- [ ] **Railway Redis:** Redis plugin provisioned; **`REDIS_URL`** + **`USE_REDIS_QUEUE=1`** on gateway and queue worker (if using async queue); worker service deployed and healthy
- [ ] **Auth:** `SUPABASE_JWT_SECRET` (and optional `SUPABASE_JWT_ISS`) on Railway gateway if Vercel app uses Supabase access tokens
- [ ] Frontend deployed to Vercel at **manthana.quaasx108.com**; **§11** env vars set (`NEXT_PUBLIC_SUPABASE_*`, **`SUPABASE_SERVICE_ROLE_KEY`**, **`ORACLE_INTERNAL_URL`**, **`NEXT_PUBLIC_APP_URL`**, proxy `NEXT_PUBLIC_*_API_URL`); `GATEWAY_CORS_ORIGINS` on Railway includes that origin
- [x] Gateway forward timeout **600s**; **502/503 retries** (up to 3) for GPU cold start
- [ ] X-ray on Modal: resolve **heatmap_url** vs split gateway (see **§15.4**) or accept `null` heatmap in UI
- [ ] Modal dashboard monitoring set up
- [ ] Backup: Keep docker-compose.yml working for local dev/fallback
- [ ] Cost alerts set in Modal dashboard ($50/mo, $100/mo thresholds)
- [ ] Railway usage / billing alerts (optional) for gateway + Redis + worker spend

---

## 13. Summary — Why This Architecture

| Aspect | Benefit |
|--------|---------|
| **Zero idle cost** | **GPU and CPU** Modal containers scale to **0** when idle — no compute charges until a request runs (see [Modal scaling](https://modal.com/docs/guide/scale) and [billing](https://modal.com/docs/guide/billing)). **Volume** weights persist on disk between cold starts (volume storage included in plan; not “per request”). |
| **One-time download** | Weights live on Modal Volume (persistent disk). Download once, reuse forever. Never re-downloaded on container start. |
| **Fast warm inference** | Within scaledown_window (60s), subsequent scans process in 5-7s with no cold start. |
| **Cost per scan** | ~$0.005-0.02 per scan in GPU cost. For 200 scans/month, that's ~$2-4 in GPU. |
| **No server management** | No VMs, no Kubernetes, no GPU driver updates. Modal handles everything. |
| **Vercel + Railway + Modal** | **Vercel** = Manthana Labs / Oracle frontend; **Railway** = gateway + report_assembly + **Railway Redis** + optional RQ worker (always-on, low fixed cost); **Modal** = **GPU** (CT/MRI, X-ray, oral **production**, etc.) + **CPU** (ECG, dermatology, optional oral CPU) inference, each **scale-to-zero**, shared **Volume** for large weights. |
| **Production ready** | Auto-scaling GPU, env-driven `*_SERVICE_URL`, `REPORT_ASSEMBLY_URL`, analyze retries; X-ray **§15**, USG **§16**, ECG **§17**, pathology / cytology / mammography **§18–§20**, lab report **§21**, dermatology **§22**, oral cancer **§23**. |

---

## 14. Complete implementation plan (executable)

This section is the **implementation spec** for engineering: what to build, in what order, and how to verify it. Repository root for paths: [`this_studio/new manthana radiology/manthana-backend`](this_studio/new%20manthana%20radiology/manthana-backend).

### 14.1 Scope

**In scope**

- CT/MRI GPU inference on Modal: `11_ct_brain`, `02_brain_mri`, `04_cardiac_ct`, `10_spine_neuro`, `08_abdominal_ct`.
- One shared Modal Volume for weights + upstream caches (TotalSegmentator, Comp2Comp, SynthSeg, Sybil, custom TorchScripts).
- Railway: API gateway + **report_assembly** (CPU) + **Railway Redis plugin** + RQ worker when `USE_REDIS_QUEUE=1`; **multipart** proxy to Modal unchanged.
- Vercel: frontend already complete — only env (`NEXT_PUBLIC_*` gateway URL, CORS).

**Out of scope (later)**

- **X-ray / ultrasound** are implemented on Modal (**§15**, **§16**); ops: deploy + set `XRAY_SERVICE_URL` / `ULTRASOUND_SERVICE_URL`.
- **ECG / dermatology:** deploy **Modal CPU** apps (**§17–§22**, `deploy_ecg.py`, `deploy_dermatology.py`) for **$0 idle**, or keep **Railway CPU** Docker if you prefer always-on; set `ECG_SERVICE_URL`, `DERMATOLOGY_SERVICE_URL`. **Oral cancer (production):** deploy **Modal GPU** `deploy_oral_cancer.py` (**§23**) and set **`ORAL_CANCER_SERVICE_URL`** to that HTTPS `…/analyze/oral_cancer` origin; use `deploy_oral_cancer_cpu.py` only as a cost alternate.
- Other modalities (pathology, etc.) on Docker or future Modal apps.
- True async job queue for CT/MRI (gateway already returns “queued” on timeout; optional hardening).
- S3 presigned uploads (add when payloads or latency require it).

### 14.2 Path and mount convention (important)

Docker Compose uses **`MODEL_DIR=/models`** and mounts the cache volume there. The shared TotalSegmentator helper uses [`shared/totalseg_runner.py`](this_studio/new%20manthana%20radiology/manthana-backend/shared/totalseg_runner.py) with `MODEL_DIR = os.getenv("MODEL_DIR", "/models")`.

**Recommendation:** Mount the Modal Volume at **`/models`** (not `/weights`) so existing code and env defaults apply. Layout example on the volume:

- `/models/ct_brain/*.pt` — TorchScripts (`CT_BRAIN_TORCHSCRIPT_PATH`, etc.)
- `/models/wmh/`, `/models/brain_lesion/` — optional MRI aux models
- `/models/SynthSeg/` or `/models/synthseg/SynthSeg/` — clone + SynthSeg weights; set `SYNTHSEG_SCRIPT` accordingly
- `/models/Comp2Comp/` — Comp2Comp clone; set `COMP2COMP_DIR`
- nnU-Net / TotalSegmentator caches under `/models` when populated by download job

Update any earlier `/weights` examples in this doc mentally to **`/models`** for consistency with the codebase.

### 14.3 Phase 0 — Accounts and local prerequisites

1. Create **Modal** workspace; install CLI (`pip install modal`); `modal token new`.
2. Create **Railway** project; note region (prefer same broad region as Modal US if possible).
3. **Vercel** project linked to frontend repo.
4. Local clone of `manthana-backend` with ability to run `modal deploy` from dev machine or CI.

**Acceptance:** `modal profile current` works; Railway and Vercel projects exist.

### 14.4 Phase 1 — Modal Volume and one-off bootstrap

1. Create volume: e.g. `modal volume create manthana-model-weights` (name is arbitrary; use consistently in code).
2. Implement [`manthana-backend/modal_app/bootstrap_weights.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/bootstrap_weights.py) (new) as a **CPU or GPU function** that:
   - Mounts volume at `/models`.
   - Installs minimal deps to run TotalSegmentator download, `git clone` Comp2Comp + optional SynthSeg, Sybil model fetch, etc.
   - Writes into `/models` and calls **`volume.commit()`** where required by Modal API.
3. Run once: `modal run modal_app/bootstrap_weights.py` (or named entrypoint).
4. Upload **proprietary** artifacts with CLI:  
   `modal volume put <volume-name> local/ich_main.pt ct_brain/ich_main.pt` (adjust paths to match env vars in secrets).
5. Create **Modal Secret** `manthana-env` with at least:  
   `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_2` (if used), `MODEL_DIR=/models`, all **`CT_BRAIN_*`**, `WMH_MODEL_PATH`, `BRAIN_LESION_MODEL_PATH`, `SYNTHSEG_SCRIPT`, `COMP2COMP_DIR`, `CT_BRAIN_DEVICE=cuda`, `TOTALSEG_DEVICE=gpu`, `DEVICE=cuda`, narrative policy vars mirroring [`docker-compose.yml`](this_studio/new%20manthana%20radiology/manthana-backend/docker-compose.yml).

**Acceptance:** Second run of bootstrap does **not** re-download large blobs; files visible under `/models` from a test Modal function.

### 14.5 Phase 2 — Five Modal applications (ASGI, minimal code churn)

**Approach:** For each service directory, expose the existing FastAPI **`app`** from [`main.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/11_ct_brain/main.py) via Modal **`modal.asgi_app()`**, not a rewrite of `inference.py`.

**New files** (under `manthana-backend/modal_app/`):

| File | Purpose |
|------|---------|
| `common.py` | Shared `modal.Volume.from_name(...)`, base CUDA image helper, `scaledown_window`, `max_containers`, secret name. |
| `deploy_ct_brain.py` | `app = modal.App(...)`; image from `services/11_ct_brain/requirements.txt` + copy `services/11_ct_brain` + `shared` into image; mount volume at `/models`; `gpu="T4"` (or L4); ASGI import `services/11_ct_brain.main:app` (adjust `PYTHONPATH` / working dir). |
| `deploy_brain_mri.py` | Heavier image: nnUNet git install per existing [`services/02_brain_mri/Dockerfile`](this_studio/new%20manthana%20radiology/manthana-backend/services/02_brain_mri/Dockerfile); **TensorFlow** for SynthSeg subprocess; `gpu="L4"`; timeout ≤ 600s. |
| `deploy_cardiac_ct.py` | Match [`services/04_cardiac_ct/Dockerfile`](this_studio/new%20manthana%20radiology/manthana-backend/services/04_cardiac_ct/Dockerfile) / `requirements.txt`; `gpu="T4"`. |
| `deploy_spine_neuro.py` | Match [`services/10_spine_neuro`](this_studio/new%20manthana%20radiology/manthana-backend/services/10_spine_neuro); `gpu="T4"`. |
| `deploy_abdominal_ct.py` | Match [`services/08_abdominal_ct`](this_studio/new%20manthana%20radiology/manthana-backend/services/08_abdominal_ct); Comp2Comp + Sybil; `gpu="L4"`; long timeout. |

**Image build notes**

- **Build context:** `manthana-backend` root so `COPY shared` and `COPY services/XX` work.
- **Ports:** irrelevant for Modal ASGI; keep FastAPI internal ports as in `config.py` — Modal fronts the app.
- **Writable uploads:** services use `/tmp/manthana_uploads`; ensure directory exists (default in code).

**Runtime options (per app)**

- `scaledown_window=60` (or 90), `min_containers=0`, `max_containers=2`–`3`.
- `secrets=[modal.Secret.from_name("manthana-env")]`.
- `allow_concurrent_inputs` / concurrency per Modal docs if you expect parallel requests per container.

**Deploy**

- `modal deploy modal_app/deploy_ct_brain.py` (repeat for each).
- Record each **HTTPS** origin Modal prints (deployment URL).

**Acceptance**

- `curl https://.../health` returns 200 for each service when secrets and weights are valid.
- `POST .../analyze/<modality>` with a small test DICOM/ZIP returns 200 and JSON shape consistent with existing `AnalysisResponse`.

### 14.6 Phase 3 — Gateway and report assembly on Railway

**Status:** Items **A**, **B**, and **D** (retries) are **implemented** in the repo. Configure Railway env only.

**A. Router —** [`manthana-backend/gateway/router.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py) (done):

- `_service_url()` helper; CT/MRI backends use `CT_BRAIN_SERVICE_URL`, `BRAIN_MRI_SERVICE_URL`, `CARDIAC_CT_SERVICE_URL`, `SPINE_NEURO_SERVICE_URL`, `ABDOMINAL_CT_SERVICE_URL` (Docker defaults if unset).
- X-ray uses `XRAY_SERVICE_URL` (same pattern as `_SERVICE_XRAY`; behaviorally equivalent to `_service_url`).

**B. Report assembly —** [`manthana-backend/gateway/main.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/main.py) (done):

- `REPORT_ASSEMBLY_URL` with `.rstrip("/")` for assemble endpoints.

**C. Railway services (production layout)**

- Deploy **gateway** from `gateway/Dockerfile` (or Nixpacks) with env vars set to Modal HTTPS **full** URLs including `/analyze/...` path.
- Deploy **report_assembly** from `services/report_assembly` (CPU); set its `OPENROUTER_*` and any paths it needs.
- **Recommended:** Deploy **ECG** and **dermatology** on **Modal CPU** (`modal deploy modal_app/deploy_ecg.py`, `deploy_dermatology.py`); deploy **oral cancer** on **Modal GPU** (`modal deploy modal_app/deploy_oral_cancer.py`) for UNI + predictable latency; point `ECG_SERVICE_URL`, `DERMATOLOGY_SERVICE_URL`, `ORAL_CANCER_SERVICE_URL` to the matching Modal HTTPS `…/analyze/...` origins (**scale to zero**). Optional: `deploy_oral_cancer_cpu.py` instead of GPU if cost dominates and UNI is unused.
- **Alternative:** Same three services as separate **Railway** CPU containers (always-on); same `*_SERVICE_URL` pattern.
- **Redis (recommended for queue):** Add **Railway → New → Redis** (managed). Set **`REDIS_URL`** on **gateway**, **queue** API (if used), and **`queue_worker`** to the **same** connection string. Set **`USE_REDIS_QUEUE=1`** where enqueue/consume happens. See **§5A (Executive E)**.
- Run **`services/queue/worker.py`** as a long-running Railway service (Dockerfile from `manthana-backend` queue profile / `services/queue`) so jobs are processed when the queue is enabled.

**D. Resilience** (done)

- Gateway analyze forward: **up to 3 attempts** with sleep on **502/503** (Modal cold start).
- Timeout **600s** for analyze forward.

**Acceptance**

- `POST /analyze` with JWT for each CT/MRI modality hits Modal and returns a complete response.
- `POST /report` succeeds when report_assembly URL is correct.

### 14.7 Phase 4 — Vercel frontend

1. Follow **§11 — Vercel — Manthana Next.js frontend** (env table for `manthana.quaasx108.com`, Supabase auth URLs, `ORACLE_INTERNAL_URL`).
2. Set **`GATEWAY_CORS_ORIGINS`** on Railway to include **`https://manthana.quaasx108.com`** (and preview URLs if needed); avoid `*` in production.
3. **Labs → gateway:** **`SUPABASE_JWT_SECRET`** on Railway ([`gateway/auth.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/auth.py)) verifies Supabase access tokens from the browser; legacy **`JWT_SECRET`** only if you still mint tokens via gateway pilot routes.

**Acceptance:** Sign-in works on production domain; Oracle chat loads; browser can upload and complete one CT and one MRI flow end-to-end without CORS errors.

### 14.8 Phase 5 — Verification matrix (go-live)

| Test | Action |
|------|--------|
| CT Brain | DICOM NCCT → `modality=ct_brain` → impression + models_used |
| Brain MRI | NIfTI or DICOM → `brain_mri` / `mri` alias |
| Cardiac | `cardiac_ct` |
| Spine | `spine_neuro` |
| Abdominal | `abdominal_ct` + optional chest region / Sybil path |
| Film photo | Gateway `film_files` (≥3 extras) → ZIP bundle → same five modalities |
| X-ray (optional) | `modality=xray` → `body_xray` on Modal or Docker; set `XRAY_SERVICE_URL`; check heatmap if split deploy (**§15.4**) |
| Ultrasound | `modality=ultrasound` (or `us` / `usg`); set `ULTRASOUND_SERVICE_URL` to Modal `…/analyze/ultrasound` (**§16**) |
| ECG | `modality=ecg`; set `ECG_SERVICE_URL` to **Modal CPU** `…/analyze/ecg` (recommended) or Railway `…/analyze/ecg` (**§17**) |
| Pathology | WSI/tiles → `modality=pathology` (or `patho` / `wsi`); set `PATHOLOGY_SERVICE_URL` (**§18**) |
| Cytology | `modality=cytology`; set `CYTOLOGY_SERVICE_URL` (**§19**) |
| Mammography | `modality=mammography`; set `MAMMOGRAPHY_SERVICE_URL`; optional **`MIRAI_HF_REPO`** (**§20**) |
| Lab report | `modality=lab_report`; set **`LAB_REPORT_SERVICE_URL`** to Modal `…/analyze/lab_report` (**§21**) |
| Dermatology | `modality=dermatology`; set **`DERMATOLOGY_SERVICE_URL`** to **Modal CPU** `…/analyze/dermatology` (recommended) or Railway (**§22**) |
| Oral cancer | `modality=oral_cancer`; set **`ORAL_CANCER_SERVICE_URL`** to **Modal GPU** **`deploy_oral_cancer.py`** (**production default**), or **Modal CPU** / Railway if using the CPU app (**§23**). For **on-service screening scores**, put **`oral_effnet_v2m.pt`** and/or **`oral_cancer_finetuned.pt`** on **`MODEL_DIR`** (default **`/models`**); default order is **V2-M before B3** when both exist (**§23**). |
| Cold start | First request after idle; note latency; second request within `scaledown_window` faster |
| Volume | No network download spike on second cold start (monitor Modal logs) |
| Cost | Modal dashboard: GPU seconds per case |

### 14.9 Risk register

| Risk | Mitigation |
|------|------------|
| SynthSeg / TensorFlow vs PyTorch in one image | Follow brain MRI Dockerfile pattern; isolate SynthSeg in subprocess (already in [`shared/synthseg_runner.py`](this_studio/new%20manthana%20radiology/manthana-backend/shared/synthseg_runner.py)). |
| Comp2Comp clone size / build time | Shallow clone in bootstrap; cache in Volume. |
| Wrong nnUNet cache path | Populate under `/models`; set env vars TotalSegmentator/nnUNet expect (verify against installed package version). |
| Modal cold start UX | Optional short “Processing…” in UI; retries in gateway; later `min_containers=1` only if budget allows. |
| JWT / CORS misconfig | Staging environment on Railway + Vercel preview first. |
| Pathology WSI / large tile batches OOM | Reduce max tiles in service config or use larger Modal GPU; persist HF caches on Volume (`/models`). |
| Mirai OOM on T4 | Switch mammography Modal function to **A10G** / **L4** (**§20.2**). |
| MedGemma OOM on T4 | Switch lab_report Modal function to **A10G** / **L4** (**§21.2**). |
| UNI / HF download fails for oral histopath | Set **`HF_TOKEN`** on service; cache under **`/models`** (Modal Volume or Railway volume); optional CPU Railway if Modal not used (**§23**). |
| Oral clinical scores mostly from cloud | Upload task-specific **`.pt`** weights to the volume (**§23.2**, **`WEIGHTS.md`**); use **`ORAL_PREFER_V2M=false`** only if B3 must run before V2-M when both files exist. |

### 14.10 Definition of Done

- All targeted modalities (including lab_report, dermatology, and oral_cancer) callable from production frontend through Railway gateway.
- Weights persist on Modal Volume; no full re-download on each container start.
- GPU scales to zero when idle; billing observable in Modal.
- Report narrative path works via `REPORT_ASSEMBLY_URL`.
- If async queue is in scope: **Railway Redis** reachable from gateway + worker; `USE_REDIS_QUEUE=1` and jobs complete without Redis connection errors.
- If Manthana Labs uses Supabase: gateway **`SUPABASE_JWT_SECRET`** set; analyze requests with `Authorization: Bearer` succeed (401s resolved).
- `docker-compose` still runs CT/MRI locally for regression (optional CI smoke on CPU dummy paths where applicable).

---

## 15. X-ray (body_xray) — production alignment and GPU

### 15.1 Is a GPU required for X-ray?

**Not strictly required, but recommended for production parity with the current code.**

- **Chest pipeline:** [shared/txrv_utils.py](this_studio/new%20manthana%20radiology/manthana-backend/shared/txrv_utils.py) loads TorchXRayVision DenseNets and uses **CUDA when `torch.cuda.is_available()`**, otherwise CPU. Two ensembles (`densenet121-res224-all` + `chex` or `mimic_nb` fallback) run per chest study — **CPU is valid but much slower** and increases request latency on a small Railway container.
- **Non-chest regions** (abdomen, bone, spine, skull): [pipeline_abdomen.py](this_studio/new%20manthana%20radiology/manthana-backend/services/01_body_xray/pipeline_abdomen.py) / [pipeline_bone.py](this_studio/new%20manthana%20radiology/manthana-backend/services/01_body_xray/pipeline_bone.py) use **OpenCV/numpy only** — no GPU needed for those paths.
- **Practical recommendation:** Run **body_xray** on **Modal with a small GPU (e.g. T4)** for the same “pay per second / scale to zero” model as CT/MRI, **or** keep a single GPU service on Railway/Docker if you colocate. Pure CPU-only hosting for the **whole** service is possible but a product/SLA choice.

### 15.2 Weights and disk (X-ray)

- TXRV checkpoints are **small** (~tens of MB per head; ~90 MB total for typical two-model ensemble + fallback). The service [Dockerfile](this_studio/new%20manthana%20radiology/manthana-backend/services/01_body_xray/Dockerfile) already **pre-downloads** weights at image build — **no Modal Volume is mandatory** for X-ray (unlike TotalSegmentator). If you mirror caches onto a Volume, **verify** the directory `torchxrayvision` uses for your pinned version rather than assuming `TORCH_HOME` alone.

### 15.3 Gateway triage vs Railway CPU (corrected nuance)

- Default **`XRAY_TRIAGE_POLICY=always_deep`** ([docker-compose.yml](this_studio/new%20manthana%20radiology/manthana-backend/docker-compose.yml)) means the gateway **does not run** `_triage_xray` or load TXRV **weights** for triage.
- Previously, **`import triage`** still imported **`txrv_utils`** at module load, pulling **PyTorch** onto the gateway at startup. **Implemented fix:** [gateway/triage.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/triage.py) **lazy-imports** `txrv_utils` **inside `_triage_xray` only**, so CPU-only Railway gateways stay light when policy is `always_deep`.
- If you set **`XRAY_TRIAGE_POLICY`** to something other than `always_deep`, the gateway process **will** run TXRV triage — still best on a machine with **GPU** if you want low latency; CPU triage is possible but heavy.

### 15.4 Heatmaps when X-ray runs on Modal (split deploy)

- [shared/heatmap_generator.py](this_studio/new%20manthana%20radiology/manthana-backend/shared/heatmap_generator.py) writes PNGs under `HEATMAP_DIR` and returns **`/heatmaps/{job_id}_heatmap.png`**.
- **Docker Compose:** gateway and `body_xray` share **`upload_data:/tmp/manthana_uploads`**, and the gateway mounts **`/heatmaps`** — URLs work.
- **Railway gateway + Modal X-ray:** the PNG is created **inside the Modal container**; the **gateway’s** `StaticFiles` directory does **not** see that file → clients may get **404** on `heatmap_url` unless you **null `heatmap_url`**, return **base64**, upload to **object storage**, or **proxy** (product choice — document in frontend).
- **Implemented default for Modal image:** [`service_image_body_xray()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) sets **`XRAY_HEATMAP_URL_MODE=none`** so the API returns **`heatmap_url: null`** and **`heatmap_type: none`** for chest (no wasted PNG I/O). For Docker Compose with a shared upload volume, leave this unset so `/heatmaps/...` keeps working.

### 15.5 Router and env (X-ray)

- **`XRAY_SERVICE_URL`** — full URL including path, e.g. `https://….modal.run/analyze/xray` or `http://body_xray:8001/analyze/xray` ([router.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py)) via **`_service_url("XRAY_SERVICE_URL", …)`**, same pattern as CT/MRI.

### 15.6 Modal app for X-ray (**in repo**)

- **Deploy:** from `manthana-backend`: `modal deploy modal_app/deploy_body_xray.py` → app id **`manthana-body-xray`**.
- **Image:** [`modal_app/common.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) `service_image_body_xray()` — CUDA 12.4 runtime, `01_body_xray/requirements.txt`, `with_manthana_llm_stack`, `shared` + service tree at `/app`, TXRV DenseNet weights pre-download (`all`, `chex`, `mimic_nb`), image env **`XRAY_HEATMAP_URL_MODE=none`** so `heatmap_url` is null when the gateway cannot serve Modal-local files.
- **Function:** [`modal_app/deploy_body_xray.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_body_xray.py) — `gpu="T4"`, Volume `/models` (parity with other services; optional for TXRV), `manthana-env` secret, `gpu_function_kwargs` timeout **600s**, scaledown **90s**.
- **Hygiene:** **`ultralytics`** removed; **`scikit-image`** pinned; **`main.py`** docstring updated; [router.py](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py) uses **`_service_url("XRAY_SERVICE_URL", …)`** like other backends.

---

## 16. Ultrasound (09_ultrasound) — Modal GPU

**Stack:** Same as X-ray/CT: **Vercel** → **Railway gateway** → **Modal GPU** for Rad-DINO inference. Optional narrative uses OpenRouter role **`narrative_usg`** ([`config/cloud_inference.yaml`](this_studio/config/cloud_inference.yaml) — `openai/gpt-4o-mini` with Llama fallback).

### 16.1 Router and env

- **`ULTRASOUND_SERVICE_URL`** — full URL including path, e.g. `https://….modal.run/analyze/ultrasound` or `http://ultrasound:8009/analyze/ultrasound` ([`gateway/router.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py)) via **`_service_url("ULTRASOUND_SERVICE_URL", …)`**.
- Legacy **`USG_SERVICE_HOST` / `USG_SERVICE_PORT`** are removed; use the single URL env only.

### 16.2 GPU and model

- Backbone: **`microsoft/rad-dino`** ([`services/09_ultrasound/inference.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/09_ultrasound/inference.py)); uses **CUDA when available**, else CPU; synthetic fallback if HF load fails.
- **Production:** deploy on **Modal T4** (or similar) for latency; **`MANTHANA_MODEL_CACHE`** is set in the Modal image and **weights are pre-downloaded at image build** in [`service_image_ultrasound()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) so cold start avoids HuggingFace round-trips.

### 16.3 Modal app (**in repo**)

- **Deploy:** from `manthana-backend`: `modal deploy modal_app/deploy_ultrasound.py` → app id **`manthana-ultrasound`**.
- **Function:** [`modal_app/deploy_ultrasound.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_ultrasound.py) — `gpu="T4"`, Volume `/models`, `manthana-env` secret, timeout **600s**, scaledown **90s**, same pattern as [`deploy_body_xray.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_body_xray.py).

### 16.4 Railway gateway

Set **`ULTRASOUND_SERVICE_URL`** to the Modal HTTPS URL printed after deploy (must include **`/analyze/ultrasound`**).

---

## 17. ECG (13_ecg) — Modal CPU (recommended) or Railway CPU

**Stack:** **Vercel** → **Railway gateway** → **ECG FastAPI** on **Modal CPU** (scale-to-zero, **$0 idle**) or **Railway** (always-on). **No GPU** and **no Modal volume / no downloaded weights** — **Manthana-ECG-Engine** (heuristic rhythm scores) + **neurokit2** intervals + OpenRouter **`narrative_ecg`** (prompt in `prompts/ecg_system.md`). Photo **quality gate** + **digitiser adapter** (OpenCV by default; optional `ECG_DIGITISER_REPO_ROOT`). See [`services/13_ecg/WEIGHTS.md`](this_studio/new%20manthana%20radiology/manthana-backend/services/13_ecg/WEIGHTS.md). SSOT LLM: [`cloud_inference.yaml`](this_studio/config/cloud_inference.yaml).

### 17.1 Modal app (**recommended for cost**)

- **Deploy:** from `manthana-backend`: `modal deploy modal_app/deploy_ecg.py` → **`manthana-ecg`**.
- **Image:** [`service_image_ecg_cpu()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) — Debian slim + `13_ecg/requirements.txt` (**no torch**), `with_manthana_llm_stack`, shared + service; **`DEVICE=cpu`**. **No** `models_volume()` — nothing to upload for ECG.
- **Autoscaler:** [`cpu_function_kwargs()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) — default **scale to zero** after idle (`scaledown_window=60`); you pay only for **CPU core-seconds** while handling traffic ([Modal billing](https://modal.com/docs/guide/billing)).

### 17.2 Router and env

- **`ECG_SERVICE_URL`** — full URL including path, e.g. `https://….modal.run/analyze/ecg` (Modal) or `https://your-ecg.up.railway.app/analyze/ecg` / `http://ecg:8013/analyze/ecg` ([`gateway/router.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py)).
- **`OPENROUTER_API_KEY`** (Modal secret **`manthana-env`**) — for narrative; rhythm + intervals still return without it.
- **`GET /health`** — **`component_health`**: `ecg_pipeline_version`, `ecg_branch`, `ecg_dl_weights: none`.

### 17.3 Docker image with LLM stack (**Railway / Compose**)

- [`services/13_ecg/Dockerfile`](this_studio/new%20manthana%20radiology/manthana-backend/services/13_ecg/Dockerfile) remains a **minimal** CPU image (local experiments).
- **Production / Compose / Railway:** use **[`Dockerfile.railway`](this_studio/new%20manthana%20radiology/manthana-backend/services/13_ecg/Dockerfile.railway)** with **build context = `this_studio`** (repo root), so the image includes **`packages/manthana-inference`**, **`config/cloud_inference.yaml`**, and **`shared/`** — required for [`llm_router`](this_studio/new%20manthana%20radiology/manthana-backend/shared/llm_router.py) and **`narrative_ecg`**.
- [`docker-compose.yml`](this_studio/new%20manthana%20radiology/manthana-backend/docker-compose.yml) **`ecg`** service uses this build; set **`OPENROUTER_API_KEY`** (and optional **`OPENROUTER_API_KEY_2`**) on the ECG service.

### 17.4 Gateway triage

- [`gateway/triage.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/triage.py) uses **`_triage_default`** for **`ecg`** — no extra GPU models on the gateway.

### 17.5 Verification

- `GET /health` — `component_health` shows `ecg_pipeline_version` / `ecg_branch` / `ecg_dl_weights`.
- `POST /analyze` with `modality=ecg` and a sample CSV/PNG reaches the ECG service; narrative fields populate when OpenRouter keys are set; without keys, core rhythm/interval output still returns.
- Response **`structures.ecg_timing_ms`**, **`ecg_pipeline_version`**, **`ecg_founder_scores`** (always `{}`).

---

## 18. Pathology (05_pathology) — Modal GPU

**Stack:** **Vercel** → **Railway gateway** → **Modal GPU** (recommended for WSI tile embedding throughput). Optional narrative: OpenRouter **`vision_pathology`** ([`cloud_inference.yaml`](this_studio/config/cloud_inference.yaml) — `openai/gpt-4o-mini`).

### 18.1 Models and aggregation

- Tile embeddings: [`shared/tile_embedding.py`](this_studio/new%20manthana%20radiology/manthana-backend/shared/tile_embedding.py) — **transformers** `paige-ai/Virchow` (via [`LazyModel`](this_studio/new%20manthana%20radiology/manthana-backend/shared/model_loader.py)) → **timm** `hf-hub:paige-ai/Virchow` → **UNI2-h** → **convnext_base.fb_in22k_ft_in1k** fallback. **`timm>=1.0`** is listed in service [`requirements.txt`](this_studio/new%20manthana%20radiology/manthana-backend/services/05_pathology/requirements.txt).
- Slide scores: [`shared/dsmil_aggregator.py`](this_studio/new%20manthana%20radiology/manthana-backend/shared/dsmil_aggregator.py) (`dsmil_slide_scores`).

### 18.2 Router and env

- **`PATHOLOGY_SERVICE_URL`** — full URL, e.g. `https://….modal.run/analyze/pathology` or `http://pathology:8005/analyze/pathology` ([`gateway/router.py`](this_studio/new%20manthana%20radiology/manthana-backend/gateway/router.py)).
- **`HF_TOKEN` / `HUGGINGFACE_TOKEN`** — recommended on Modal secret and Docker for gated or reliable HF pulls (Virchow / UNI2).

### 18.3 Docker / Compose

- Production image: **[`services/05_pathology/Dockerfile.railway`](this_studio/new%20manthana%20radiology/manthana-backend/services/05_pathology/Dockerfile.railway)** (context **`this_studio`**) — OpenSlide + `manthana-inference` + `cloud_inference.yaml` + `shared/`. [`docker-compose.yml`](this_studio/new%20manthana%20radiology/manthana-backend/docker-compose.yml) **`pathology`** service uses this build.

### 18.4 Modal app (**in repo**)

- **Deploy:** `modal deploy modal_app/deploy_pathology.py` → **`manthana-pathology`**.
- **Image:** [`service_image_pathology()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) — CUDA, OpenSlide apt packages, ConvNeXt **warm-up** at image build to shorten first fallback path; **`MODEL_DIR`** / **`MANTHANA_MODEL_CACHE`** → `/models` (Modal Volume).

---

## 19. Cytology (11_cytology) — Modal GPU

**Stack:** Same as pathology: embeddings + DSMIL + OpenRouter role **`cytology`** in [`cloud_inference.yaml`](this_studio/config/cloud_inference.yaml).

### 19.1 Router and env

- **`CYTOLOGY_SERVICE_URL`** — `https://….modal.run/analyze/cytology` or `http://cytology:8011/analyze/cytology`.

### 19.2 Docker / Compose

- **[`Dockerfile.railway`](this_studio/new%20manthana%20radiology/manthana-backend/services/11_cytology/Dockerfile.railway)** (context **`this_studio`**). Compose **`cytology`** service updated like pathology.

### 19.3 Modal app (**in repo**)

- **Deploy:** `modal deploy modal_app/deploy_cytology.py` → **`manthana-cytology`**.

---

## 20. Mammography (12_mammography) — Modal GPU

**Stack:** **Mirai** from Hugging Face — repo id **`MIRAI_HF_REPO`** (default **`Lab-Rasool/Mirai`**) per [`services/12_mammography/inference.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/12_mammography/inference.py). Optional narrative: **`mammography`** role in [`cloud_inference.yaml`](this_studio/config/cloud_inference.yaml) (`gpt-4o-mini`).

### 20.1 Router and env

- **`MAMMOGRAPHY_SERVICE_URL`** — `https://….modal.run/analyze/mammography` or `http://mammography:8012/analyze/mammography`.
- **`MIRAI_HF_REPO`** — override Mirai checkpoint repo if needed.
- **`HF_TOKEN`** — often required for Mirai snapshot download.

### 20.2 GPU sizing

- Default Modal **`gpu="T4"`** in [`deploy_mammography.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_mammography.py). If Mirai hits **OOM**, switch function decorator to **`A10G`** (or **L4**) in that file and redeploy.

### 20.3 Docker / Compose

- **[`Dockerfile.railway`](this_studio/new%20manthana%20radiology/manthana-backend/services/12_mammography/Dockerfile.railway)**; compose passes **`MIRAI_HF_REPO`**, HF tokens.

### 20.4 Modal app (**in repo**)

- **Deploy:** `modal deploy modal_app/deploy_mammography.py` → **`manthana-mammography`**.

---

## 21. Lab report (15_lab_report) — Modal GPU (recommended)

**Stack:** Structured parsing via [`shared/medical_document_parser.py`](this_studio/new%20manthana%20radiology/manthana-backend/shared/medical_document_parser.py) (**`google/medgemma-4b-it`**, lazy load, ~9GB VRAM in docs). Clinical interpretation: OpenRouter role **`lab_report`** in [`cloud_inference.yaml`](this_studio/config/cloud_inference.yaml) (`meta-llama/llama-3.3-70b-instruct` with fallbacks).

### 21.1 Router and env

- **`LAB_REPORT_SERVICE_URL`** — `https://….modal.run/analyze/lab_report` or `http://lab_report:8015/analyze/lab_report`.
- **`OPENROUTER_API_KEY`** (Modal secret + service).
- **`HF_TOKEN`** / **`HUGGINGFACE_TOKEN`** — useful when MedGemma is pulled from Hugging Face (ModelScope is tried first in code).

### 21.2 GPU sizing

- Default Modal **`gpu="T4"`** in [`deploy_lab_report.py`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/deploy_lab_report.py). If MedGemma hits **OOM**, switch to **`A10G`** (or **L4**) in that file and redeploy.

### 21.3 Docker / Compose

- **[`Dockerfile.railway`](this_studio/new%20manthana%20radiology/manthana-backend/services/15_lab_report/Dockerfile.railway)** (context **`this_studio`**). GPU compose reserves **NVIDIA**; `docker-compose.cpu.yml` can run the same image with **`DEVICE=cpu`** (slow for MedGemma).

### 21.4 Modal app (**in repo**)

- **Deploy:** `modal deploy modal_app/deploy_lab_report.py` → **`manthana-lab-report`**.
- **Image:** [`service_image_lab_report()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) — CUDA runtime, lab `requirements.txt`, `with_manthana_llm_stack`, shared + service copy; **`MODEL_DIR`** / **`MANTHANA_MODEL_CACHE`** → `/models` (Modal Volume).

### 21.5 Smoke

```bash
curl -sS "https://YOUR_MODAL_LAB_URL/health"
curl -sS "https://YOUR_MODAL_LAB_URL/ready"
```

---

## 22. Dermatology (16_dermatology) — Modal CPU (recommended) or Railway CPU

**Stack (production upgrade):** **HAM10000-style 7-class** scores from **torchvision EfficientNet-V2-M** when **`derm_efficientnet_v2m_ham7.pt`** (or **`DERM_HAM_CHECKPOINT`**) is on disk; else legacy **EfficientNet-B4** + **`derm_efficientnet_b4.pt`**; else **OpenRouter vision JSON** for the same **`DERM_CLASSES`** keys. Order is **`DERM_CLASSIFIER_PRIORITY`** (default `ham_v2,b4,openrouter`). **Narrative** always uses OpenRouter role **`dermatology`** in [`cloud_inference.yaml`](this_studio/config/cloud_inference.yaml). Raw seven-class probs are returned as **`structures.ham10000_scores`** when the HAM branch runs.

**Runbook / provenance / class order / rollback:** [`services/16_dermatology/WEIGHTS.md`](this_studio/new%20manthana%20radiology/manthana-backend/services/16_dermatology/WEIGHTS.md). **Implementation:** [`analyzer.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/16_dermatology/analyzer.py), [`ham_classifier.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/16_dermatology/ham_classifier.py), [`ham_map.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/16_dermatology/ham_map.py).

### 22.1 Modal app (**recommended for cost**)

- **Deploy:** `modal deploy modal_app/deploy_dermatology.py` → **`manthana-dermatology`**.
- **Image:** [`service_image_dermatology_cpu()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py); **Volume** `/models` for optional weights; **`DEVICE=cpu`**; [`cpu_function_kwargs()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) for **scale to zero**. If **EfficientNet-V2-M + optional Grad-CAM** OOMs, raise **memory** (e.g. **6144–8192 MiB**) in `deploy_dermatology.py`.

### 22.2 Router and env

- **`DERMATOLOGY_SERVICE_URL`** — `https://….modal.run/analyze/dermatology` (Modal) or `https://your-derm.railway.app/analyze/dermatology` / `http://dermatology:8016/analyze/dermatology`.
- **`OPENROUTER_API_KEY`**, optional **`OPENROUTER_API_KEY_2`**.
- **`MODEL_DIR`** — mount or volume; optional **`derm_efficientnet_v2m_ham7.pt`** (HAM7) and/or **`derm_efficientnet_b4.pt`** (12-class). Without weights, scores come from OpenRouter vision JSON.
- **`DERM_CLASSIFIER_PRIORITY`** — e.g. `ham_v2,b4,openrouter` or `openrouter` only.
- **`DERM_HAM_CHECKPOINT`** — HAM weight filename under `MODEL_DIR` (default **`derm_efficientnet_v2m_ham7.pt`**).
- **`DERM_HAM7_CLASS_ORDER`** — seven comma-separated keys matching training logits (default **`akiec,bcc,bkl,df,mel,nv,vasc`**).
- **`DERM_GRADCAM`** — `1` / `true` for optional **`structures.derm_gradcam_png_base64`** (non-blocking).

**Observability:** **`models_used`** lists the score branch (e.g. **`EfficientNet-V2-M-HAM7`**, **`EfficientNet-B4-derm`**, **`openrouter-vision-derm-scores`**) plus the narrative model slug. **`GET /ready`** and **`GET /health`** include **`component_health`** (weight presence, resolved mode, priority).

### 22.3 Weights — Modal Volume / Railway

Upload checkpoints to the shared Modal volume under **`/models`**. Example (from **`manthana-backend`**):

```bash
modal volume put manthana-model-weights ./path/to/derm_efficientnet_v2m_ham7.pt derm_efficientnet_v2m_ham7.pt
modal volume put manthana-model-weights ./path/to/derm_efficientnet_b4.pt derm_efficientnet_b4.pt
```

**Rollback:** remove or rename a bad checkpoint; the service falls through to the next priority tier.

### 22.4 Docker / Compose (**Railway alternative**)

- **[`Dockerfile.railway`](this_studio/new%20manthana%20radiology/manthana-backend/services/16_dermatology/Dockerfile.railway)** (context **`this_studio`**, **python:3.11-slim** + OpenGL libs for PIL/torchvision).

### 22.5 Smoke

```bash
curl -sS "https://YOUR_DERM_URL/health"
curl -sS "https://YOUR_DERM_URL/ready"
```

---

## 23. Oral cancer (14_oral_cancer) — Modal GPU (**production default**) / Modal CPU (cost option) / Railway

**Production:** Point **`ORAL_CANCER_SERVICE_URL`** at **`modal deploy modal_app/deploy_oral_cancer.py`** ( **`manthana-oral-cancer`**, **T4**, shared volume **`/models`**, **8 GiB** container RAM for HF/UNI cache). **`manthana-env`** should include **`OPENROUTER_API_KEY`**, **`ORAL_CANCER_ENABLED=true`**, **`HF_TOKEN`** when using UNI, and **`MODEL_DIR=/models`**.

**Stack (production upgrade):** **Screening-style class probabilities** for clinical photos come from **on-service vision weights** first; cloud LLM is **fallback JSON + narrative**, not the sole score source.

- **Clinical photo:** **torchvision EfficientNet-V2-M** + **`oral_effnet_v2m.pt`** (preferred when the file exists) and/or **`google/efficientnet-b3`** (Transformers) + **`oral_cancer_finetuned.pt`**. **Default order** when **both** weight files are on disk: **V2-M before B3** (override with **`ORAL_PREFER_V2M`** — see below). Supports **3-class** heads (Normal / OPMD / OSCC) or **2-class** (normal vs malignant) via **`ORAL_V2M_NUM_CLASSES=2`** and a documented softmax → three-bucket mapping (**`ORAL_V2M_BINARY_OPMD_FRACTION`**).
- **Histopath / H&E-style:** **`MahmoodLab/UNI`** (`UNI_MODEL_ID`) + optional **`uni_oral_linear_head.pt`** (unchanged).
- **Cloud:** OpenRouter role **`oral_cancer`** in [`cloud_inference.yaml`](this_studio/config/cloud_inference.yaml) (primary model **`moonshotai/kimi-k2.5:online`** at time of writing) for structured vision fallback and long-form narrative when local classifiers are missing or degrade.

**Runbook / provenance / class order / rollback:** [`services/14_oral_cancer/WEIGHTS.md`](this_studio/new%20manthana%20radiology/manthana-backend/services/14_oral_cancer/WEIGHTS.md). **Implementation:** [`inference.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/14_oral_cancer/inference.py), [`config.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/14_oral_cancer/config.py).

### 23.1 Router and env

- **`ORAL_CANCER_SERVICE_URL`** — **`https://….modal.run/analyze/oral_cancer`** from **`deploy_oral_cancer.py`** (GPU, **recommended**), or the CPU app / Railway if you deliberately choose lower cost (**one URL only**).
- **`MODEL_DIR`** — **`/models`** on Modal (shared volume); must contain uploaded **`.pt`** files you intend to use (V2-M and/or B3 checkpoint).
- **`OPENROUTER_API_KEY`**, optional **`OPENROUTER_API_KEY_2`**; optional **`CLOUD_INFERENCE_CONFIG_PATH`** if the SSOT YAML is not at the default mount.
- **`HF_TOKEN`** / **`HUGGINGFACE_TOKEN`** — often needed for UNI snapshot download.
- **`ORAL_CANCER_ENABLED`**, **`DEVICE`** / **`ORAL_DEVICE`** (compose), optional **`UNI_MODEL_ID`**, **`UNI_HEAD_CHECKPOINT`**.
- **Classifier preference & V2-M head (see [`config.py`](this_studio/new%20manthana%20radiology/manthana-backend/services/14_oral_cancer/config.py)):** **`ORAL_PREFER_V2M`** — unset = V2-M before B3 when both files exist; **`false`** or **`0`** = legacy **B3-first**; **`true`** or **`1`** = force V2-M first when V2-M weights exist. **`ORAL_EFFNET_V2M_CHECKPOINT`** — filename under `MODEL_DIR` (default **`oral_effnet_v2m.pt`**). **`ORAL_V2M_NUM_CLASSES`** — **`2`** or **`3`**. **`ORAL_V2M_BINARY_OPMD_FRACTION`** — when `NUM_CLASSES=2`, fraction of malignant mass mapped to OPMD (default **`0.45`**).

**Observability:** response **`models_used`** lists which branch ran (e.g. **`EfficientNet-V2-M`**, **`EfficientNet-B3`**, **`UNI`**, **`openrouter-vision-oral`**). **`GET /health`** / **`component_health`** includes oral weight flags and resolved classifier order when the service loads [`get_loaded_status`](this_studio/new%20manthana%20radiology/manthana-backend/services/14_oral_cancer/inference.py).

### 23.2 Weights — Modal Volume / Railway

- Upload **`oral_effnet_v2m.pt`** (and optionally **`oral_cancer_finetuned.pt`**, **`uni_oral_linear_head.pt`**) to the shared Modal volume at paths under **`/models`** (same pattern as other modalities). Example (from **`manthana-backend`**):

```bash
modal volume put manthana-model-weights ./path/to/oral_effnet_v2m.pt oral_effnet_v2m.pt
```

- Short reference and env table: **[`modal_app/README.md`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/README.md)** (oral subsection) and **`WEIGHTS.md`** above.
- **Rollback:** remove or rename a bad checkpoint on the volume; the service falls through to the other EfficientNet branch if present, then OpenRouter vision JSON if configured.

### 23.3 Docker / Compose (**Railway alternative**)

- **[`Dockerfile.railway`](this_studio/new%20manthana%20radiology/manthana-backend/services/14_oral_cancer/Dockerfile.railway)** (context **`this_studio`**, **python:3.11-slim** + GL libs). Mount **`model_cache`** at **`/models`** for checkpoints and HF cache.

### 23.4 Modal app — **GPU** (**production default**)

- **Deploy:** `modal deploy modal_app/deploy_oral_cancer.py` → **`manthana-oral-cancer`**.
- **Image:** [`service_image_oral_cancer()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) — CUDA runtime, oral `requirements.txt`, `with_manthana_llm_stack`, shared + service; EfficientNet-B3 **processor** warmed at image build; **`MODEL_DIR`** / **`MANTHANA_MODEL_CACHE`** → `/models`.
- Default **`gpu="T4"`**, **`memory=8192`** (MiB) for stable UNI/HF loads; switch to **A10G** / **L4** if UNI + concurrent load OOMs on GPU.
- **Runtime defaults** in the Modal stub: **`DEVICE=cuda`**, **`ORAL_CANCER_ENABLED=true`**, **`MODEL_DIR=/models`**.

### 23.5 Modal app — **CPU** (cost / low-volume **alternate**)

- **Deploy:** `modal deploy modal_app/deploy_oral_cancer_cpu.py` → **`manthana-oral-cancer-cpu`**.
- **Image:** [`service_image_oral_cancer_cpu()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py) — CPU slim image, oral `requirements.txt`, Volume **`/models`**, **`DEVICE=cpu`**, [`cpu_function_kwargs()`](this_studio/new%20manthana%20radiology/manthana-backend/modal_app/common.py). Runs **V2-M and/or B3** on CPU when weights are present. **Do not** point production **`ORAL_CANCER_SERVICE_URL`** here if you rely on **UNI** histopath latency or GPU throughput.

### 23.6 Smoke

```bash
curl -sS "https://YOUR_ORAL_URL/health"
curl -sS "https://YOUR_ORAL_URL/ready"
```
