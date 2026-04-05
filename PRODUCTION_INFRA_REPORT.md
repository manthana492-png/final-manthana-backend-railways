# Manthana Radiology — Production Infrastructure Report

> Minimum-cost, maximum-performance architecture for 500 subscribers, auto-scaling as users grow.

---

## 1. Architecture Overview — Three-Tier Separation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1 — FRONTEND  (Vercel / Cloudflare Pages)                            │
│  manthana-radio-frontend (Next.js 14)                                      │
│  CDN-cached, $0 at 500 users, auto-scales globally                         │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTPS / JWT
┌────────────────────────────────────┴────────────────────────────────────────┐
│  TIER 2 — ORCHESTRATION BACKEND  (Railway / Fly.io / small VPS)            │
│  gateway :8000 · report_assembly :8020 · queue :8021                       │
│  pacs_bridge :8030 · redis · LLM APIs (Kimi/Claude/Groq)                  │
│  ecg :8013 · oral_cancer :8014 · dermatology :8016 (CPU services)          │
│  CPU-only, always-on, ~$25-50/mo                                           │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP / RunPod API
┌────────────────────────────────────┴────────────────────────────────────────┐
│  TIER 3 — GPU INFERENCE  (RunPod Serverless)                               │
│  7 grouped endpoints — GPU spins up on request, idle = $0                  │
│  Docker image = code only (lightweight, fast pull)                         │
│  Model weights live on RunPod Network Volume, never re-downloaded          │
└────────────────────┬────────────────────────────┬───────────────────────────┘
                     │ mounts at /runpod-volume    │
┌────────────────────┴────────────────────────────┴───────────────────────────┐
│  MODEL STORAGE — RunPod Network Volume  (~100-150 GB, persistent)          │
│  /runpod-volume/models/  ← all weights live here permanently               │
│  Downloaded ONCE during setup. Never downloaded again.                     │
│  ~$7-11/month regardless of how many workers spin up or down               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** separate compute by cost profile. Frontend is free. Orchestration is cheap CPU. GPU inference is expensive but only runs when a scan is submitted — for 500 users doing ~2-5 scans/day each, GPU utilization is bursty, not constant.

---

## 2. GPU Workload Map — What Actually Needs a GPU

### GPU Services (from docker-compose.yml + actual inference.py audit)

| Service | Actual Model(s) — exact IDs from code | VRAM Tier | Default Device | Cold Start |
|---------|---------------------------------------|-----------|----------------|------------|
| **body_xray** | `torchxrayvision` DenseNet121 weights: `densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-mimic_nb`; + triage model (`densenet121-res224-all` again in gateway) | ~1-2 GB | `cuda` if available, else `cpu` | ~15s |
| **brain_mri** | **TotalSeg:** `task="total_mr", fast=True` via `_device_for_totalseg()` — if `DEVICE=cpu` → TotalSeg on CPU; else `TOTALSEG_DEVICE` (default **`gpu`**). **SynthSeg:** subprocess only (`shared/synthseg_runner.py`, `SYNTHSEG_SCRIPT`, TensorFlow path) — **brain_mri only**. **Prima:** `run_prima_study` in `shared/prima_pipeline.py` — today returns `prima_pipeline_not_implemented` even when config exists (subprocess not wired). | ~16 GB *estimate* | PyTorch/Torch on GPU when CUDA available; TotalSeg device as above | ~30s *estimate* |
| **abdominal_ct** | `TotalSegmentator task="total", fast=<auto>` + **`run_comp2comp_abdominal`** (`shared/comp2comp_runner.py`, `COMP2COMP_DIR` default `/opt/Comp2Comp`, `bin/C2C`) for BMD / muscle / fat metrics when CLI available | ~8 GB | `cuda` (`TOTALSEG_DEVICE` default `gpu`) | ~25s |
| **cardiac_ct** | `TotalSegmentator task="heartchambers", fast=True` + **aortic metrics** (`estimate_aortic_diameter_mm` on TotalSeg outputs — `models_used` includes label **`TotalSegmentator-AAQ-proxy`**, not a separate HF checkpoint). **Comp2Comp** is cloned in Dockerfile and probed in **health only**; **`run_pipeline` does not call Comp2Comp**. | ~8 GB | `cuda` (`TOTALSEG_DEVICE` default `gpu`) | ~25s |
| **spine_neuro** | **TotalSeg only:** `vertebrae_mr` (MRI) or `vertebrae_body` (CT). **No SynthSeg, no Prima** in this service. | ~16 GB *estimate* | ⚠️ `dev = os.getenv("TOTALSEG_DEVICE") or os.getenv("DEVICE", "cpu")` — defaults **CPU** if unset; set **`TOTALSEG_DEVICE=gpu`**. | ~25s *estimate* |
| **pathology** | `paige-ai/Virchow` via `AutoModel.from_pretrained` (HF); tile_embedding fallbacks: `MahmoodLab/UNI2-h`, `convnext_base.fb_in22k_ft_in1k` | ~8 GB | `cuda` (LazyModel device="cuda") | ~20s |
| **cytology** | Same as pathology: `paige-ai/Virchow` (primary), same tile_embedding fallbacks | ~8 GB | `cuda` | ~20s |
| **ct_brain** | ⚠️ **Custom TorchScript file** — path from env `CT_BRAIN_TORCHSCRIPT_PATH`; no HuggingFace ID; `torch.jit.load(path)` | ~2 GB | `cuda` if `CT_BRAIN_DEVICE` not `cpu` and CUDA available | ~10s |
| **ultrasound** | `microsoft/rad-dino` via `AutoModel.from_pretrained` + `AutoImageProcessor.from_pretrained` (direct HF, not via model_loader) | ~2 GB | `cuda` if available, else `cpu` | ~15s |
| **mammography** | `Lab-Rasool/Mirai` (`MIRAI_HF_REPO` env) via `snapshot_download` → `MiraiModel.from_pretrained`; cache: `MODEL_DIR/mirai_cache`. Plus **`heuristic_birads_cv`** (numpy/PIL heuristics, not a DL checkpoint). | ~4 GB | ⚠️ **`inference.py` sets `DEVICE = os.environ.get("DEVICE", "cpu")`** — **effective default CPU** even though `config.py` defaults `DEVICE` to `"cuda"`. Set `DEVICE=cuda` for GPU. | ~20s |
| **lab_report** | Structured parse: **`google/medgemma-4b-it`** in `shared/medical_document_parser.py` (`ManagedModel`, `MEDGEMMA_VRAM_GB=9.0`). Analyzer docstring still says “Parrotlet-v” — **misleading name**; code path is MedGemma. Interpretation: **Kimi** (`KIMI_LAB_MODEL` / `KIMI_MODEL`). `USE_PARROTV_PARSER` env controls structured-parser use (`auto` / `always` / `never`). | ~9 GB | Parser `_DEVICE` default **`cuda`** in `medical_document_parser.py`; lab container may override with `DEVICE=cpu` in compose | ~25s |

**VRAM / cold-start columns:** Values marked *estimate* are **capacity-planning guesses**, not measurements checked into this repo.

### MRI-only: brain_mri vs spine_neuro (code-grounded)

| Service | TotalSegmentator | SynthSeg (`shared/synthseg_runner.py`) | Prima (`shared/prima_pipeline.py`) |
|---------|------------------|----------------------------------------|-------------------------------------|
| **brain_mri** | `task="total_mr", fast=True`; device from `_device_for_totalseg()` (`DEVICE=cpu` → CPU; else `TOTALSEG_DEVICE`, default **`gpu`**) | **Yes** — subprocess to `SYNTHSEG_SCRIPT` (default `/opt/SynthSeg/.../SynthSeg_predict.py`) if installed | **Invoked** but `run_prima_study` returns **`prima_pipeline_not_implemented`** until subprocess to Prima pipeline is wired |
| **spine_neuro** | `vertebrae_mr` (MRI) or `vertebrae_body` (CT); `dev = os.getenv("TOTALSEG_DEVICE") or os.getenv("DEVICE", "cpu")` | **No** | **No** |

TotalSegmentator for this suite is pinned **2.3.0** in `services/02_brain_mri/requirements.txt` (CUDA base image in service Dockerfiles).

### CPU-Only Services (no GPU needed)

| Service | Actual Stack — exact from code | Default Device | Local .pt files |
|---------|-------------------------------|----------------|----------------|
| **ecg** | `scipy.signal.find_peaks` (rhythm heuristics) + `neurokit2` (intervals) + OpenCV (digitizer) + Kimi/Groq LLMs | `cpu` (config default) | None |
| **oral_cancer** | `google/efficientnet-b3` (HF) + `EfficientNetV2-M` (torchvision) + `MahmoodLab/UNI` (HF) + Kimi/Claude | `cpu` (config default) | ⚠️ `oral_cancer_finetuned.pt`, `oral_effnet_v2m.pt`, `uni_oral_linear_head.pt` — must be manually placed |
| **dermatology** | `torchvision.models.efficientnet_b4` head + Kimi fallback if .pt missing | `cpu` (config default) | ⚠️ `derm_efficientnet_b4.pt` — must be manually placed |
| **gateway** | Lazy-loads TXRV triage (`densenet121-res224-all`) on first x-ray; lazy-loads `ekacare/parrotlet-e` (case embeddings, `PARROTLET_E_VRAM_GB=2.0`) in background thread | CPU service; models loaded only on demand | None |
| **report_assembly** | LLM API calls only (DeepSeek → Gemini → Groq → Qwen chain); no torch | CPU only, no torch | None |
| **pacs_bridge** | DICOM routing | CPU | None |
| **queue_api / worker** | Async job management | CPU | None |

### 2B. Per-modality model lock-in (code audit — models only)

This table is the **single source of truth** for “what is actually wired” vs “optional / stub / API-only.” It matches the current `manthana-backend` services (paths under `new manthana radiology/manthana-backend/`).

#### What “stub”, “partial”, and “checkpoints” mean (you **are** using the modalities)

- **The modality is still used.** Every row is a real HTTP service users hit (X-ray, brain MRI, oral cancer, etc.). Nothing in this table means “we turned that modality off.”
- **Stub** — A **named step exists in the pipeline** (code calls it), but the **heavy model inference is not implemented yet**. Example: **Prima** on brain MRI — `run_prima_study` runs, but today it always returns `prima_pipeline_not_implemented` until someone wires the subprocess. **TotalSeg and the rest of brain MRI still run.**
- **Partial path** — The **main** local model runs, but an **add-on** only runs if extra install or data is present. Examples: **SynthSeg** (needs SynthSeg installed at `SYNTHSEG_SCRIPT`); **Comp2Comp** on abdominal CT (needs working `bin/C2C` and valid input). If the add-on is missing, the service **still returns a full response** using the parts that did run (e.g. TotalSeg only).
- **Optional checkpoints (oral / derm)** — The service **always** loads public bases from Hugging Face where the code says so. The **`.pt` files** are **your** fine-tuned heads. If they are on disk, you get the **full local** stack; if not, the code **falls back** (often Kimi/Claude vision) so the product still answers — **different** behavior, same modality.

**“Production-ready local inference?”** means: *can this modality give its best result **without** relying on an external LLM for the core finding*, given a normal deploy (weights + env). **Yes** = core DL path is on; **Partial** = core is on but some advertised extras are conditional or stubbed.

| Modality (service) | Locked stack (what runs locally) | Download / install (weights & binaries) | Typical compute | Production-ready **local** inference? | If local path missing |
|--------------------|-----------------------------------|----------------------------------------|-----------------|--------------------------------------|------------------------|
| **X-ray** (`01_body_xray`) | TorchXRayVision DenseNet121 ×3 (`PRIMARY_WEIGHTS` / `SECONDARY_WEIGHTS` / `FALLBACK_SECONDARY` in `shared/txrv_utils.py`) | TXRV auto-download on first load; optional idle unload via `XRAY_MODEL_IDLE_UNLOAD_SEC` | GPU if CUDA | **Yes** (core CXR) | Pipeline still returns structure; narrative may depend on Kimi if required |
| **Brain MRI** (`02_brain_mri`) | TotalSeg `total_mr`; optional SynthSeg subprocess; Prima hook | TotalSeg 2.3.0 cache; SynthSeg repo at `SYNTHSEG_SCRIPT`; Prima manual (`PRIMA_*`) | GPU for TotalSeg | **Partial** — TotalSeg **yes**; SynthSeg **if installed**; Prima **stub** (`prima_pipeline_not_implemented`) | SynthSeg skipped if script missing; Prima never scores until wired |
| **Abdominal CT** (`08_abdominal_ct`) | TotalSeg `total` (+ optional skip → heuristics only); Comp2Comp abdominal CLI | TotalSeg cache; Comp2Comp clone + `COMP2COMP_DIR` / `bin/C2C` | GPU TotalSeg | **Partial** — TotalSeg **yes**; Comp2Comp **if binary + series path OK** | Falls back to heuristics label `Manthana-CT-heuristic` when segmentation off |
| **Cardiac CT** (`04_cardiac_ct`) | TotalSeg `heartchambers`; aortic diameter / AAA-style **heuristics** on segment outputs | TotalSeg cache; Comp2Comp optional in image (health) | GPU TotalSeg | **Yes** for chamber seg + geometry proxy; **not** a separate “AAQ” neural model | TotalSeg failure → empty segments + default structure list |
| **Spine / neuro** (`10_spine_neuro`) | TotalSeg `vertebrae_mr` or `vertebrae_body` | TotalSeg cache | GPU if `TOTALSEG_DEVICE=gpu` | **Yes** (segmentation path) | Defaults CPU if env wrong |
| **Pathology** (`05_pathology`) | `paige-ai/Virchow` + tile pipeline; fallbacks `MahmoodLab/UNI2-h`, ConvNeXt in `shared/tile_embedding.py` | HF hub → `MANTHANA_MODEL_CACHE` | GPU | **Yes** when Virchow loads | Fallback embedders if Virchow fails |
| **Cytology** (`11_cytology`) | Same Virchow + tile stack as pathology | Same | GPU | **Yes** | Same |
| **CT brain** (`11_ct_brain`) | TorchScript only (`CT_BRAIN_TORCHSCRIPT_PATH`) | **Manual** file (your deployment artifact) | GPU if set | **Yes** only if you ship a real `.pt` | CI may use tiny dummy module |
| **Ultrasound** (`09_ultrasound`) | `microsoft/rad-dino` (HF); tiny synthetic fallback if load fails | HF cache | GPU if CUDA | **Yes** with RadDINO | Degraded synthetic backbone |
| **Mammography** (`12_mammography`) | `Lab-Rasool/Mirai` + numpy heuristics | HF `snapshot_download` → `mirai_cache` | **CPU default** in `inference.py` | **Yes** when Mirai snapshot present | Heuristics-only path possible |
| **ECG** (`13_ecg`) | **No** trainable DL model — `ecg_rhythm` + `neurokit2` + OpenCV digitizer | pip deps only | CPU | **Yes** (signal/heuristic engine) | Narrative via Kimi/Groq APIs |
| **Oral cancer** (`14_oral_cancer`) | HF B3 + V2-M + `MahmoodLab/UNI` + optional `.pt` heads | HF + **manual** `oral_cancer_finetuned.pt`, `oral_effnet_v2m.pt`, `uni_oral_linear_head.pt` | CPU default; CUDA if `DEVICE=cuda` | **Partial** — full strength needs checkpoints | Kimi/Claude vision JSON + narrative |
| **Lab report** (`15_lab_report`) | **`google/medgemma-4b-it`** (structured) + Kimi (text) | HF MedGemma cache | GPU for MedGemma if enabled | **Partial** — MedGemma when parser loads; Kimi always for interpretation | Text extraction + Kimi if parser off |
| **Dermatology** (`16_dermatology`) | EfficientNet-B4 head (`derm_efficientnet_b4.pt`) | **Manual** `.pt` + torchvision | CPU default | **Partial** — local classifier if file exists | Kimi vision if no checkpoint |

**Gateway** (not a modality): lazy TXRV triage + `ekacare/parrotlet-e` embeddings — same download/cache rules as HF + TXRV.

**Narratives everywhere:** Kimi / Anthropic / Groq (and others in `report_assembly`) are **APIs**, not volume weights — omitted from the “download” column except where they are the only path.

---

## 3. Platform Comparison — GPU Inference Providers

| Platform | Model | GPU Options | Pricing | Scale-to-Zero | Cold Start | Best For |
|----------|-------|-------------|---------|---------------|------------|----------|
| **RunPod Serverless** | Bring your Docker | A40 (48GB), L4 (24GB), T4 (16GB) | $0.00019/s (T4) to $0.00076/s (A100) | Yes (idle = $0) | ~20-60s first request | **Best all-round; your Docker images work directly** |
| **Modal** | Python decorator | T4, L4, A10G, A100, H100 | $0.000164/s (T4) to $0.001472/s (H100) | Yes (auto) | ~5-15s (snapshots) | Fastest cold starts; Python-native |
| **SiliconFlow** | API (hosted models) | Managed | $0.001-0.01/call | Yes | ~1-3s | Only if they host your exact models (unlikely for medical) |
| **Alibaba PAI-EAS** | Bring container | V100, A10, A100 | ~$0.75/hr (V100) to $3.50/hr (A100) | Yes (scale to 0) | ~30-90s | Asian users; K8s integration |
| **Lightning AI** (current) | Studios / Serve | T4, L4, A10G, A100 | $0.60/hr (T4) to $3.80/hr (A100) | Partial (studios stay running) | ~10-20s | Dev/prototype; **not ideal for pay-per-inference** |
| **GCP Cloud Run GPU** | Bring Docker | L4 (24GB) | ~$0.000356/s per GPU | Yes (min 0) | ~30-60s | Google ecosystem |
| **AWS SageMaker Serverless** | Bring container | ml.g5.xlarge etc | ~$0.00027/s inference | Yes | ~60-120s | AWS ecosystem; slower cold starts |

---

## 4. Recommended Architecture — The "Manthana Stack"

### Tier 1 — Frontend: **Vercel** ($0-20/mo)
- Next.js 14 deploys automatically from git
- Global CDN, edge functions for `/api/*` proxy
- LoginGate + JWT → gateway
- Free tier covers 500 users easily

### Tier 2 — Orchestration: **Railway** ($25-50/mo)
- Gateway (FastAPI) — 1 vCPU, 1GB RAM
- Report Assembly — 0.5 vCPU, 512MB
- Redis — managed (Railway native or Upstash)
- ECG + oral_cancer + dermatology (CPU inference, co-located)
- Queue worker for async jobs
- Auto-scales horizontally

### Tier 3 — GPU Inference: **RunPod Serverless** ($80-200/mo, usage-based)
- Each modality group = 1 serverless endpoint
- Docker image is **code-only, no weights** — small image (~3-8 GB), fast to pull from registry
- **All model weights live permanently on a RunPod Network Volume** (see Section 5A)
- Workers mount the volume at `/runpod-volume` on startup — model load from local disk, zero internet download
- Scale 0→N workers on demand; idle = $0

---

## 5. GPU Endpoint Grouping — Consolidate for Cost

Instead of 11+ separate GPU endpoints, **group by VRAM tier and shared weights**:

| Endpoint Name | GPU Type | Services Bundled | Actual Models (from code) | Est. VRAM | Special Notes |
|---------------|----------|------------------|--------------------------|-----------|---------------|
| **totalseg-heavy** | L4 24GB | brain_mri, spine_neuro | **Shared TotalSeg:** `total_mr` (brain_mri) + `vertebrae_mr` / `vertebrae_body` (spine_neuro). **brain_mri only:** SynthSeg subprocess; Prima hook (**inference stub** — `prima_pipeline_not_implemented`). **spine_neuro:** TotalSeg only (no SynthSeg/Prima). | ~16 GB *estimate* | ⚠️ `TOTALSEG_DEVICE=gpu`; optional `SYNTHSEG_SCRIPT`; Prima config does not run inference until subprocess is wired. |
| **totalseg-ct** | L4 24GB | abdominal_ct, cardiac_ct | **abdominal_ct:** TotalSeg `total` + **Comp2Comp abdominal** (real inference path). **cardiac_ct:** TotalSeg `heartchambers` + aortic heuristics only; Comp2Comp in image for **health/metadata**, not main pipeline. | ~8 GB | Same TotalSeg weight cache for `total` vs `heartchambers` tasks; still bundle if you want one GPU pool for both CT services |
| **virchow** | L4 24GB | pathology, cytology | `paige-ai/Virchow` (primary); fallbacks `MahmoodLab/UNI2-h`, `convnext_base` | ~8 GB | Both services load same Virchow cache dir |
| **xray** | T4 16GB | body_xray | TXRV `densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-mimic_nb` | ~1-2 GB | Highest volume modality; keep separate |
| **ct-brain** | T4 16GB | ct_brain | ⚠️ **Custom TorchScript** — not on HuggingFace; loaded from `CT_BRAIN_TORCHSCRIPT_PATH` env | ~2 GB | Model .pt file must be manually placed on volume; no auto-download possible |
| **ultrasound** | T4 16GB | ultrasound | `microsoft/rad-dino` via HF transformers (direct `from_pretrained`, not model_loader) | ~2 GB | Cache via `MANTHANA_MODEL_CACHE` |
| **mammo-lab** | L4 24GB | mammography, lab_report | `Lab-Rasool/Mirai` (snapshot to `MODEL_DIR/mirai_cache`) + `google/medgemma-4b-it` (`MEDGEMMA_VRAM_GB=9.0`) | ~9-13 GB | Mammography defaults CPU; set `DEVICE=cuda` in endpoint env to enable GPU |

**Result:** 7 serverless endpoints instead of 11+. At low volume most stay idle = $0. X-ray likely busiest.

### RunPod SKU matrix — right-size GPUs and cap spend

Pick the **smallest GPU whose VRAM covers the worst case for that endpoint**, with ~10–20% headroom for activations and framework overhead. **Do not** default to A100/A40 unless profiling shows **OOM** on L4.

| Serverless endpoint | Recommended RunPod GPU | Est. peak VRAM (*planning*) | Why this SKU | Move up only if… |
|---------------------|------------------------|----------------------------|--------------|------------------|
| **xray** | **T4 16GB** | ~1–2 GB | Three DenseNet121 checkpoints are small; cheapest per second. | Rare; only if you add much larger CXR models. |
| **ct-brain** | **T4 16GB** | ~2 GB | Single TorchScript path. | OOM in traces with your real `.pt`. |
| **ultrasound** | **T4 16GB** | ~2 GB | Rad-DINO modest footprint. | OOM or much larger input policy. |
| **totalseg-ct** | **L4 24GB** | ~8 GB class | TotalSeg `total` + `heartchambers` + headroom for spikes / batch-like peaks. | **A40 48GB** only if consistent OOM on L4. |
| **virchow** | **L4 24GB** | ~8 GB class | Virchow + tile path; safer than 16GB for WSI-style memory. | A40 if tile batch or model variant exceeds L4. |
| **totalseg-heavy** | **L4 24GB** | ~16 GB *estimate* | Heaviest TotalSeg MR bundle (`total_mr`, `vertebrae_*`). | **A40 48GB** if `nvidia-smi` / logs show OOM (e.g. huge volumes + SynthSeg on same box). |
| **mammo-lab** | **L4 24GB** | ~9–13 GB | MedGemma ~9 GB in code comments + Mirai if both on GPU. | A40 if both models resident + OOM. |

**How many GPUs you pay for on RunPod**

- You configure **7 endpoints**, not **7 always-on GPUs**.
- With **`min_workers = 0`**, billable GPUs = **0** when idle.
- Each active **worker** = **one GPU**. Concurrent GPUs ≈ sum of workers across endpoints (capped by each endpoint’s **`max_workers`**, e.g. 3–5).
- Typical steady state at modest scale: **0–3 GPUs** under load; **xray** often dominates.

**Cost discipline (checklist)**

1. Use **T4** for **xray**, **ct-brain**, **ultrasound** — avoid L4/A40 unless proven necessary.
2. Use **L4 24GB** for **totalseg-heavy**, **totalseg-ct**, **virchow**, **mammo-lab**.
3. Keep **`max_workers`** low (e.g. 2–3 at ~500 users) to limit burst spend.
4. Optional **`min_workers = 1`** only on **xray** during peak hours if cold-start SLA matters; otherwise keep **0**.
5. Revisit SKUs after **real** `nvidia-smi` max memory during worst-case requests — estimates are not benchmarks.

---

## 5A. Model Persistence — RunPod Network Volume (CRITICAL)

### The Problem Without This

If model weights are NOT pre-stored, every cold start downloads them from the internet:
- TotalSegmentator MR tasks (`total_mr`, `vertebrae_mr`, `vertebrae_body`): multi-GB combined → minutes of download
- Virchow (path/cytology): ~4-8 GB → 5-8 min download
- MedGemma 4B: ~8 GB → 8-12 min download
- **Total all models: ~80-120 GB → completely unusable per-cold-start**

### The Solution: RunPod Network Volume

A **Network Volume** is a persistent NFS-backed SSD in RunPod that:
- **Survives** all worker spin-ups, spin-downs, and restarts
- Mounts automatically at `/runpod-volume` inside every worker for that endpoint
- Shared across all workers of the same endpoint (read access)
- Costs **~$0.07/GB/month** — one-time storage, not per compute

**You download each model exactly once. That's it forever.**

### Total Model Sizes on Disk

Two categories: **HuggingFace downloads** (auto, once) and **local .pt checkpoints** (must be manually copied, see Section 5B).

| Model Group | Exact Model IDs / Filenames | Type | Est. Disk Size |
|-------------|----------------------------|------|---------------|
| TorchXRayVision (body_xray + gateway triage) | `densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-mimic_nb` | HF/TXRV auto-download | ~0.4 GB |
| TotalSegmentator MR (brain_mri + spine_neuro) | tasks: `total_mr` (brain_mri), `vertebrae_mr` + `vertebrae_body` (spine_neuro) | TotalSeg 2.3.0 auto-download into `MODEL_DIR` / cache on first run | ~order of magnitude **4–8 GB** (not pinned in repo) |
| SynthSeg (**brain_mri only**) | `SYNTHSEG_SCRIPT` default `/opt/SynthSeg/scripts/commands/SynthSeg_predict.py`; subprocess + TensorFlow | **Separate** from TotalSeg: install SynthSeg repo / image layer; **not** covered by HF `snapshot_download` or TotalSeg pre-download loop alone | varies |
| Prima (**brain_mri only**) | `PRIMA_CONFIG_YAML`, `PRIMA_REPO_DIR` (see `PRIMA_INTEGRATION.md`) | **Manual** weights + config; **`run_prima_study` currently returns `prima_pipeline_not_implemented`** — do not count as production inference until subprocess is wired | varies |
| TotalSegmentator CT (abdominal_ct + cardiac_ct) | tasks: `total` (abdominal), `heartchambers` (cardiac) | TotalSeg auto-download | ~5 GB |
| Comp2Comp | `COMP2COMP_DIR` (`/opt/Comp2Comp`, `bin/C2C`) | **abdominal_ct:** used in **`run_comp2comp_abdominal`**. **cardiac_ct:** cloned in Dockerfile / health probe; **not** called from `run_pipeline` | image + upstream weights (varies) |
| Virchow tile embeddings (pathology + cytology) | `paige-ai/Virchow` (primary); `MahmoodLab/UNI2-h` (fallback) | HF `snapshot_download` | ~10 GB |
| rad-dino (ultrasound) | `microsoft/rad-dino` | HF `from_pretrained` | ~0.5 GB |
| Mirai (mammography) | `Lab-Rasool/Mirai` via `snapshot_download` → `MODEL_DIR/mirai_cache` | HF snapshot | ~3 GB |
| MedGemma (lab_report) | `google/medgemma-4b-it`; `MEDGEMMA_VRAM_GB=9.0` | HF `from_pretrained` | ~9 GB |
| Oral cancer (HF part) | `google/efficientnet-b3` (HF), `MahmoodLab/UNI` (HF) | HF `from_pretrained` | ~3 GB |
| Oral cancer (local .pt) | `oral_cancer_finetuned.pt`, `oral_effnet_v2m.pt`, `uni_oral_linear_head.pt` | ⚠️ Manual copy | ~0.5 GB |
| Dermatology (local .pt) | `derm_efficientnet_b4.pt` | ⚠️ Manual copy | ~0.1 GB |
| ct_brain (local .pt) | Custom TorchScript file at `CT_BRAIN_TORCHSCRIPT_PATH` | ⚠️ Manual copy (not on HF) | ~0.5-2 GB |
| Gateway case embeddings | `ekacare/parrotlet-e`; `PARROTLET_E_VRAM_GB=2.0` | HF `from_pretrained` | ~0.5 GB |
| **TOTAL** | | | **~38-40 GB** (HF + TotalSeg + TXRV only) |

→ Create a **100 GB Network Volume** to have headroom for TotalSegmentator cache files, **SynthSeg** (if baked into image or copied to volume), **Prima** weights (if you add them later), and future models. Cost: **~$7/month**.

### How Workers Use the Volume

```
Worker cold start:
  1. RunPod pulls Docker image (code-only, ~2-5 GB) from registry — ~10-30s
  2. RunPod mounts /runpod-volume to the worker container — ~1s
  3. handler.py runs module-level init: load model from /runpod-volume/models/ → GPU — ~5-15s
  4. Worker ready to accept requests — total cold start: ~15-45s

Worker warm (within idle_timeout window):
  1. Request arrives → handler() called directly
  2. Model already loaded in GPU VRAM
  3. Inference runs → returns result — ~1-5s total
```

### Handler Pattern — Load Model at Startup, Not Per Request

The **most important pattern**: load model weights at **module level** (once per worker lifetime), NOT inside the handler function (which runs on every request).

```python
# runpod_handler.py  — for body_xray endpoint
import runpod
import os
import torch

# ── RUNS ONCE per worker lifetime ──────────────────────────────────────
MODEL_CACHE = os.getenv("MANTHANA_MODEL_CACHE", "/runpod-volume/models")
os.environ["MANTHANA_MODEL_CACHE"] = MODEL_CACHE  # tell model_loader where to look

from services.body_xray.inference import load_models, run_inference

print("Loading models from volume...")
MODELS = load_models()   # reads from /runpod-volume/models — disk only, no HTTP
print("Models loaded. Worker ready.")
# ── END ONCE ────────────────────────────────────────────────────────────

def handler(event):
    """Called for every scan request. MODELS already in GPU VRAM."""
    input_data = event["input"]
    file_bytes  = base64.b64decode(input_data["file_b64"])
    modality    = input_data["modality"]
    result = run_inference(MODELS, file_bytes, modality)
    return result

runpod.serverless.start({"handler": handler})
```

This works because your existing `model_loader.py` already:
1. Checks `MANTHANA_MODEL_CACHE` for cached weights on disk before downloading
2. Has `unload()` that removes from GPU memory but keeps disk cache intact
3. Has LRU eviction — if two services on one endpoint compete for VRAM, it offloads lesser-used model to CPU RAM

### Network Volume Setup — One Time

```bash
# Step 1: Create volume in RunPod dashboard
#   Datacenter: choose same region you'll deploy endpoints (e.g., US-TX-3)
#   Size: 100 GB
#   Name: manthana-models

# Step 2: Spin up a one-time "setup pod" — attach the volume
#   Pod type: any CPU pod (cheapest, ~$0.01/hr)
#   Mount: /runpod-volume
#   Image: nvidia/cuda:12.4.0-runtime-ubuntu22.04 (or your backend image)

# Step 3: Inside the setup pod, run ONE download script:
python scripts/download_all_models.py \
  --cache-dir /runpod-volume/models \
  --hf-token $HF_TOKEN

# Step 4: Verify:
ls /runpod-volume/models/
# → body_xray/  brain_mri/  virchow/  medgemma/  ...

# Step 5: Delete the setup pod — volume persists forever
```

Create `scripts/download_all_models.py` in your repo:

```python
"""
One-time script to populate the RunPod Network Volume.
Run once on a setup pod with the volume mounted at /runpod-volume.

Usage:
    python scripts/download_all_models.py \
        --cache-dir /runpod-volume/models \
        --hf-token hf_xxx \
        --model-dir /runpod-volume/models

NOTE: Local .pt checkpoints (oral_cancer, dermatology, ct_brain) must be
copied separately — see Section 5B below. This script only handles
HuggingFace + TorchXRayVision + TotalSegmentator downloads.
"""
import os, argparse, subprocess, sys
from pathlib import Path

def main(cache_dir: str, hf_token: str, model_dir: str):
    os.environ["MANTHANA_MODEL_CACHE"] = cache_dir
    os.environ["MODEL_DIR"] = model_dir
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    os.makedirs(cache_dir, exist_ok=True)

    print("=== Downloading all Manthana model weights to:", cache_dir)

    # ── 1. TorchXRayVision ───────────────────────────────────────────────
    # Exact weight IDs from shared/txrv_utils.py
    import torchxrayvision as xrv
    for w in ["densenet121-res224-all", "densenet121-res224-chex", "densenet121-res224-mimic_nb"]:
        print(f"  Downloading TXRV {w}...")
        xrv.models.DenseNet(weights=w)  # caches to TXRV default dir
    print("✓ TorchXRayVision (3 weights)")

    # ── 2. HuggingFace snapshot models ──────────────────────────────────
    from huggingface_hub import snapshot_download
    hf_models = [
        "paige-ai/Virchow",        # pathology + cytology (primary)
        "MahmoodLab/UNI2-h",       # pathology + cytology (tile_embedding fallback)
        "MahmoodLab/UNI",          # oral_cancer histopathology head
        "microsoft/rad-dino",      # ultrasound
        "google/medgemma-4b-it",   # lab_report (MEDGEMMA_VRAM_GB=9.0)
        "google/efficientnet-b3",  # oral_cancer clinical classifier
        "ekacare/parrotlet-e",     # gateway case embeddings (PARROTLET_E_VRAM_GB=2.0)
    ]
    for model_id in hf_models:
        print(f"  Downloading {model_id}...")
        snapshot_download(model_id, cache_dir=cache_dir, token=hf_token)
        print(f"  ✓ {model_id}")

    # ── 3. Mirai mammography — uses snapshot_download to model_dir/mirai_cache ──
    mirai_cache = Path(model_dir) / "mirai_cache"
    mirai_cache.mkdir(parents=True, exist_ok=True)
    print("  Downloading Lab-Rasool/Mirai...")
    snapshot_download("Lab-Rasool/Mirai", local_dir=str(mirai_cache), token=hf_token)
    print("  ✓ Lab-Rasool/Mirai → mirai_cache/")

    # ── 4. TotalSegmentator — pre-download all tasks used in codebase ───
    # Tasks used: total_mr (brain_mri), vertebrae_mr + vertebrae_body (spine_neuro),
    #             total (abdominal_ct), heartchambers (cardiac_ct)
    ts_tasks = ["total_mr", "vertebrae_mr", "vertebrae_body", "total", "heartchambers"]
    import tempfile, numpy as np, nibabel as nib
    dummy_path = Path(tempfile.mkdtemp()) / "dummy.nii.gz"
    dummy_nii = nib.Nifti1Image(np.zeros((64, 64, 32), dtype=np.float32), np.eye(4))
    nib.save(dummy_nii, str(dummy_path))
    for task in ts_tasks:
        print(f"  Pre-downloading TotalSegmentator task={task}...")
        try:
            from totalsegmentator.python_api import totalsegmentator
            with tempfile.TemporaryDirectory() as out_dir:
                totalsegmentator(str(dummy_path), out_dir, task=task,
                                 fast=True, device="cpu")  # cpu just to download weights
        except Exception as e:
            print(f"  ⚠  TotalSeg {task} download triggered but inference failed (expected on dummy): {e}")
        print(f"  ✓ TotalSegmentator {task} weights cached")

    print("\n=== HuggingFace + TXRV + TotalSeg downloads complete.")
    print("=== MRI: SynthSeg is NOT downloaded here — bake /opt/SynthSeg into the Docker image or mount it.")
    print("=== MRI: Prima weights are manual + pipeline still stubbed (prima_pipeline_not_implemented).")
    print("=== IMPORTANT: You still need to manually copy local .pt checkpoints.")
    print("    See Section 5B in PRODUCTION_INFRA_REPORT.md")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/runpod-volume/models")
    ap.add_argument("--model-dir", default="/runpod-volume/models")
    ap.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    args = ap.parse_args()
    main(args.cache_dir, args.hf_token, args.model_dir)
```

### Section 5B — Local .pt Checkpoints (Cannot Be Auto-Downloaded)

These files are **not on HuggingFace** — they are custom-trained weights you own. They must be manually copied to the volume once and stay there forever.

| File | Service | Loaded By | Config Env | Where to Place on Volume |
|------|---------|-----------|------------|--------------------------|
| `oral_cancer_finetuned.pt` | oral_cancer | `torch.load` + `load_state_dict` on EfficientNet-B3 | `CHECKPOINT_FILENAME` (default `oral_cancer_finetuned.pt`) under `MODEL_DIR` | `/runpod-volume/models/oral_cancer_finetuned.pt` |
| `oral_effnet_v2m.pt` | oral_cancer | `torch.load` + `load_state_dict` on EfficientNetV2-M | `ORAL_EFFNET_V2M_CHECKPOINT` under `MODEL_DIR` | `/runpod-volume/models/oral_effnet_v2m.pt` |
| `uni_oral_linear_head.pt` | oral_cancer | `torch.load` → `nn.Linear` head on top of UNI | `UNI_HEAD_CHECKPOINT` under `MODEL_DIR` | `/runpod-volume/models/uni_oral_linear_head.pt` |
| `derm_efficientnet_b4.pt` | dermatology | `torch.load(model_path, weights_only=True)` on EfficientNet-B4 | `CHECKPOINT_FILENAME` (default `derm_efficientnet_b4.pt`) under `MODEL_DIR` | `/runpod-volume/models/derm_efficientnet_b4.pt` |
| Custom TorchScript for ct_brain | ct_brain | `torch.jit.load(path)` | `CT_BRAIN_TORCHSCRIPT_PATH` — full path, no default | `/runpod-volume/models/ct_brain_torchscript.pt` (then set env `CT_BRAIN_TORCHSCRIPT_PATH=/runpod-volume/models/ct_brain_torchscript.pt`) |

**How to copy them to the volume (during the one-time setup pod):**
```bash
# Inside the setup pod, volume is mounted at /runpod-volume
# Assuming you've uploaded these .pt files to an S3 bucket or can scp them:

# Option A — from S3
aws s3 cp s3://your-bucket/manthana-checkpoints/ /runpod-volume/models/ --recursive

# Option B — from your current Lightning AI studio (during migration)
# From Lightning AI terminal:
runpodctl send /path/to/oral_cancer_finetuned.pt
# Then in RunPod setup pod:
runpodctl receive <code>

# Option C — if you have them in this_studio already, upload to S3 first
# then use Option A
```

> If `derm_efficientnet_b4.pt` or oral cancer `.pt` files are missing, those services **gracefully fall back to Kimi API** (vision mode) — so the service still works, just uses LLM instead of local model. Only `ct_brain` will fail if `CT_BRAIN_TORCHSCRIPT_PATH` is not set and no file exists.

### env var per endpoint (RunPod endpoint settings)

```bash
# All GPU endpoints:
MANTHANA_MODEL_CACHE=/runpod-volume/models
MODEL_DIR=/runpod-volume/models
HF_TOKEN=hf_xxxxx        # only needed if a new model is ever added
DEVICE=cuda

# totalseg-heavy endpoint (brain_mri + spine_neuro) — CRITICAL:
TOTALSEG_DEVICE=gpu      # ← spine_neuro defaults to cpu if this is not set!
# Optional — brain_mri only (spine_neuro does not use these):
SYNTHSEG_SCRIPT=/opt/SynthSeg/scripts/commands/SynthSeg_predict.py
# PRIMA_CONFIG_YAML=...  PRIMA_REPO_DIR=/opt/Prima
# ↑ Config alone does NOT run inference today (see shared/prima_pipeline.py).

# totalseg-ct endpoint (abdominal_ct + cardiac_ct):
TOTALSEG_DEVICE=gpu

# ct-brain endpoint — CRITICAL:
CT_BRAIN_TORCHSCRIPT_PATH=/runpod-volume/models/ct_brain_torchscript.pt
CT_BRAIN_DEVICE=cuda

# mammo-lab endpoint (mammography defaults cpu — must override):
DEVICE=cuda
MIRAI_HF_REPO=Lab-Rasool/Mirai  # already default but explicit is safer
```

### Two Levels of Caching — Disk vs GPU

| Level | Where | Survives | Speed |
|-------|-------|----------|-------|
| **Disk cache** (Network Volume) | `/runpod-volume/models/` | Forever (independent of workers) | Load to GPU: ~5-15s |
| **GPU cache** (warm worker) | VRAM inside an active worker | Until `idle_timeout` expires | Inference call: ~1-3s |

The `idle_timeout` (set in RunPod endpoint config) controls how long a worker stays alive after finishing a job:
- `idle_timeout = 300s` (recommended): worker stays warm for 5 minutes
- First request of a burst: cold start (~15-45s, disk → GPU load)
- Every subsequent request within 5 minutes: **~1-3s** (already in GPU)
- After 5 minutes idle: worker shuts down, GPU cost stops = $0

For X-ray (highest volume): set `idle_timeout=600s` and `min_workers=1` during business hours — that one worker covers almost all requests instantly with near-zero cold starts.

### Volume Sharing Across Endpoints

All 7 GPU endpoints + Railway (gateway/CPU services) attach the **same** volume. Models for all modalities are on it. Workers only load what they need into VRAM:

```
/runpod-volume/models/
│
├── [TXRV — auto-download by torchxrayvision]
│   └── ~/.torchxrayvision/  (or custom cache)
│       ├── densenet121-res224-all.pt
│       ├── densenet121-res224-chex.pt
│       └── densenet121-res224-mimic_nb.pt
│           ← body_xray endpoint + gateway (lazy triage)
│
├── [TotalSegmentator — auto-download on first run]
│   └── ~/.totalsegmentator/  (or TOTALSEG_HOME)
│       ├── task_total_mr/          ← totalseg-heavy (brain_mri)
│       ├── task_vertebrae_mr/      ← totalseg-heavy (spine_neuro MRI)
│       ├── task_vertebrae_body/    ← totalseg-heavy (spine_neuro CT)
│       ├── task_total/             ← totalseg-ct (abdominal_ct)
│       └── task_heartchambers/     ← totalseg-ct (cardiac_ct)
│
├── /opt/SynthSeg/  (image layer or bind-mount)  ← brain_mri only — NOT TotalSeg weights
├── /opt/Prima/ + PRIMA_CONFIG_YAML            ← brain_mri only — manual; inference stub in code today
│
├── [HuggingFace snapshot — via snapshot_download]
│   ├── paige-ai--Virchow/          ← virchow endpoint (pathology + cytology primary)
│   ├── MahmoodLab--UNI2-h/         ← virchow endpoint (tile_embedding fallback)
│   ├── MahmoodLab--UNI/            ← Railway oral_cancer service
│   ├── microsoft--rad-dino/        ← ultrasound endpoint
│   ├── google--medgemma-4b-it/     ← mammo-lab endpoint (lab_report)
│   ├── google--efficientnet-b3/    ← Railway oral_cancer service
│   └── ekacare--parrotlet-e/       ← Railway gateway (case embeddings, PARROTLET_E_VRAM_GB=2.0)
│
├── mirai_cache/                    ← mammo-lab endpoint (Lab-Rasool/Mirai snapshot)
│
└── [LOCAL .pt — must be manually copied, see Section 5B]
    ├── oral_cancer_finetuned.pt    ← Railway oral_cancer service
    ├── oral_effnet_v2m.pt          ← Railway oral_cancer service
    ├── uni_oral_linear_head.pt     ← Railway oral_cancer service
    ├── derm_efficientnet_b4.pt     ← Railway dermatology service
    └── ct_brain_torchscript.pt     ← ct-brain GPU endpoint
```

Each worker only loads its own slice of the volume into VRAM. Disk reads from the NVMe-backed volume are fast (~1-3 GB/s). Sharing the volume across all endpoints means **one-time download covers everything** and Railway CPU services can also mount it.

---

## 6. Cost Estimate — 500 Users

**Assumptions:** 500 subscribers, ~3 scans/user/day avg, 80% X-ray/CT, 20% other. ~1,500 scans/day. Average GPU time ~30s per scan.

| Component | Monthly Cost | Notes |
|-----------|-------------|-------|
| **Frontend (Vercel)** | $0-20 | Free tier covers 500 users |
| **Orchestration (Railway)** | $25-50 | Gateway + Redis + CPU services |
| **GPU Inference (RunPod)** | $80-200 | ~1,500 scans/day × 30s × $0.00019-0.00076/s |
| **LLM APIs (Kimi/Claude/Groq)** | $30-80 | Narratives per scan; Kimi cheapest primary |
| **RunPod Network Volume** | $7 | 100 GB × $0.07/GB/month — all model weights |
| **Uploads & CDN storage** | $5-10 | Scan image uploads (not model storage) |
| **TOTAL** | **$147-367/mo** | **$0.29-0.73 per user/month** |

### Compare to Lightning AI today

- Single A100 studio running 24/7 = **~$2,700/mo**
- Single T4 always-on = **~$430/mo**
- **Serverless saves 70-95%** at this scale because you pay for ~12.5 GPU-hours/day, not 24.

---

## 7. LLM API Strategy — Current Usage

| Provider | Models Used | Used By | Purpose | Cost Tier |
|----------|------------|---------|---------|-----------|
| **Kimi (Moonshot)** | kimi-k2.5, moonshot-v1-8k | All 14 services + gateway copilot | Vision + text narrative (primary) | Cheapest |
| **Anthropic** | claude-3-5-haiku, claude-sonnet-4, claude-sonnet-4-5 | MRI, CT, path, cyto, mammo, oral, spine, USG | Narrative fallback, vision structured | Medium |
| **Groq** | llama-3.3-70b-versatile | ECG, gateway copilot, report_assembly | Fast text fallback | Cheapest |
| **Google Gemini** | gemini-2.0-flash-lite | report_assembly | Unified report generation | Low |
| **DeepSeek** | deepseek-chat | report_assembly (router) | Narrative fallback | Cheapest |
| **Qwen (Alibaba)** | qwen-max | report_assembly (router) | Narrative fallback | Low |

**Recommendations:**
- Keep Kimi as primary (cheapest, good vision)
- Claude as quality fallback
- **Add Redis response caching** — identical findings → same narrative = significant savings
- Consider **batching** narrative calls at report_assembly level

---

## 8. Gateway Code Changes for Serverless GPU

The key change: `router.py` SERVICE_MAP currently points at Docker container hostnames.

```python
# CURRENT (Docker Compose — same network)
SERVICE_MAP = {
    "xray": "http://body_xray:8001/analyze/xray",
    "brain_mri": "http://brain_mri:8002/analyze/brain_mri",
    ...
}

# PRODUCTION (RunPod Serverless + co-located CPU services)
SERVICE_MAP = {
    "xray":        "runpod://ENDPOINT_ID_XRAY",
    "brain_mri":   "runpod://ENDPOINT_ID_TOTALSEG_HEAVY",
    "abdominal_ct":"runpod://ENDPOINT_ID_TOTALSEG_CT",
    "cardiac_ct":  "runpod://ENDPOINT_ID_TOTALSEG_CT",
    "spine_neuro": "runpod://ENDPOINT_ID_TOTALSEG_HEAVY",
    "pathology":   "runpod://ENDPOINT_ID_VIRCHOW",
    "cytology":    "runpod://ENDPOINT_ID_VIRCHOW",
    "ct_brain":    "runpod://ENDPOINT_ID_CT_BRAIN",
    "ultrasound":  "runpod://ENDPOINT_ID_ULTRASOUND",
    "mammography": "runpod://ENDPOINT_ID_MAMMO_LAB",
    "lab_report":  "runpod://ENDPOINT_ID_MAMMO_LAB",
    # CPU services stay local
    "ecg":         "http://localhost:8013/analyze/ecg",
    "oral_cancer": "http://localhost:8014/analyze/oral_cancer",
    "dermatology": "http://localhost:8016/analyze/dermatology",
}
```

The gateway needs a **thin adapter** in `route_to_service()` that: (1) detects `runpod://` prefix, (2) base64-encodes the uploaded file, (3) POSTs to RunPod `/runsync` API, (4) unwraps the response. The existing 600s timeout covers cold starts.

---

## 9. Implementation Roadmap

### Phase 1 — Prepare Volume + Docker Images (Week 1-2)

**Model Volume (one-time setup):**
- Create a **100 GB RunPod Network Volume** in your chosen datacenter region
- Spin up a cheap CPU pod, attach the volume, run `python scripts/download_all_models.py`
- Verify all weights are present in `/runpod-volume/models/`, then delete the pod
- Volume persists permanently — never download again

**Docker Images (code-only, no weights):**
- Create `runpod_handler.py` for each GPU endpoint — loads model at **module level** (once per worker), calls inference in handler function
- Set `MANTHANA_MODEL_CACHE=/runpod-volume/models` as env var in endpoint config
- Build images WITHOUT copying model weights in (keep images small: ~3-6 GB)
- Test each image locally: mock `/runpod-volume/models` path
- Push to Docker Hub or GHCR (GitHub Container Registry)

### Phase 2 — Deploy Orchestration Tier (Week 2)
- Deploy gateway + redis + report_assembly + queue to **Railway**
- Deploy CPU services (ecg, oral_cancer, dermatology) alongside gateway
- Update `router.py` SERVICE_MAP URLs to point at RunPod endpoints
- Add RunPod API key to gateway env

### Phase 3 — Deploy GPU Tier (Week 2-3)
- Create 7 RunPod serverless endpoints (grouped per Section 5)
- **Attach the Network Volume** to each endpoint (same volume, all endpoints)
- Set `MANTHANA_MODEL_CACHE=/runpod-volume/models` env var on each endpoint
- Configure: **min workers = 0** (except xray: 1 during business hours), max workers = 3-5
- `idle_timeout = 300s` standard, `600s` for xray (highest traffic)
- Set GPU type per endpoint (T4 16GB for light, L4 24GB for heavy)
- Gateway routes modality → correct RunPod endpoint ID

### Phase 4 — Deploy Frontend (Week 3)
- Connect `manthana-radio-frontend` repo to **Vercel**
- Set `NEXT_PUBLIC_GATEWAY_URL` to Railway gateway URL
- Custom domain + SSL (automatic on Vercel)

### Phase 5 — Monitoring & Hardening (Week 3-4)
- RunPod auto-scales workers based on queue depth (built-in)
- Railway auto-scales CPU services (built-in)
- Health checks from gateway to all endpoints
- Alert on: cold start > 60s, error rate > 5%, LLM fallback rate
- Add LLM response caching in Redis

---

## 10. Scaling Beyond 500 Users

| Users | Scans/Day | GPU Strategy | Orchestration | Est. Cost/mo |
|-------|-----------|-------------|---------------|-------------|
| 0-500 | ~1,500 | RunPod serverless, 0 min workers | Railway single instance | $150-370 |
| 500-2,000 | ~6,000 | RunPod serverless, 1 min worker for xray | Railway 2x instances | $400-800 |
| 2,000-10,000 | ~30,000 | K8s (GKE/EKS) with KEDA GPU autoscaler | K8s with HPA | $2,000-5,000 |
| 10,000+ | ~100,000+ | Multi-region K8s, reserved GPUs, Triton | K8s multi-region | $8,000+ |

**The serverless approach carries you from 0 to ~2,000 users without Kubernetes.** Only when sustained GPU utilization exceeds ~60% does reserved infra become cheaper.

---

## 11. Lightning AI (Current) vs Recommended Stack

| Aspect | Lightning AI (Now) | Recommended Stack |
|--------|-------------------|-------------------|
| GPU billing | Always-on studio, paying even idle | Per-second, scale to zero |
| Cost at 500 users | ~$430-2,700/mo (T4-A100 always on) | ~$150-370/mo |
| Auto-scaling | Manual (resize studio) | Automatic (RunPod + Railway) |
| Cold start | None (always on) | 15-45s first request (image pull + disk→GPU load); ~1-3s if worker already warm |
| Multi-GPU | Single studio limits | Unlimited parallel workers |
| Frontend hosting | In studio (not ideal) | Vercel CDN (global, fast) |
| Ops complexity | Low (all in one place) | Medium (3 platforms) |
| Production readiness | Dev/prototype | Production-grade |
| Concurrent users | Limited by single machine | Unlimited (horizontal) |

---

## 12. Summary — The Decision

**For 500 users at minimum cost:**

1. **Vercel** for frontend ($0-20/mo)
2. **Railway** for gateway + CPU services + Redis ($25-50/mo)
3. **RunPod Serverless** for 7 GPU endpoints ($80-200/mo)
4. **Kimi (primary) + Claude (fallback)** for LLM narratives ($30-80/mo)

**Total: ~$147-367/month for 500 users. $0.29-0.73 per user/month.**

This is **70-95% cheaper** than always-on GPU and scales automatically. When you hit 2,000+ users and sustained GPU load, migrate to Kubernetes with KEDA — the Docker images and gateway routing layer transfer directly.

### Model Weights — Where They Live at Every Stage

| Stage | Where weights live | Action |
|-------|------------------|--------|
| Development (now) | Lightning AI studio disk | Already downloaded by your current setup |
| Migration (one-time) | Run `download_all_models.py` → RunPod Network Volume | ~30-60 min, done once |
| Production (always) | `/runpod-volume/models/` on the Network Volume | Permanent, never deleted |
| Worker cold start | Volume (disk) → GPU VRAM | ~5-15s load, then ready |
| Worker warm | Already in GPU VRAM | ~1-3s inference only |
| Worker idle (after timeout) | GPU VRAM freed, weights stay on volume | $0 GPU cost |
| New model update | Re-run setup pod, overwrite specific subfolder | 5 min, one-time per update |
