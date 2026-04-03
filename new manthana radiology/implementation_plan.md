# Manthana v4 — Complete Implementation Plan

**All 6 Engine Architectures · File-by-File · Codebase-Aligned**

> **Core Principle:** All models download on GPU at runtime via HuggingFace Hub. Nothing is baked into Docker images. The [LazyModel](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py#19-105) → `ManagedModel` upgrade preserves this pattern.

---

## Phase Execution Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
  FDA       Dual     Cascade    Memory    Correlation  Registry
  Layer     CXR      Inference  Manager   Engine       Hot-Swap
  ─────     ────     ─────────  ───────   ──────────   ────────
  ~3 days   ~2 days  ~2 days    ~2 days   ~2 days      ~2 days
```

**Total estimated effort: ~13 days sequential**

---

## Phase 1: FDA Foundation Layer

> Wire TotalSegmentator, Comp2Comp (FDA 510(k)), and MONAI as the backbone for all CT/MRI services.

### Proposed Changes

---

#### Shared Infrastructure

##### [MODIFY] [model_loader.py](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py)

**What changes:** Add a `ManagedModel` class that extends [LazyModel](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py#19-105) with VRAM tracking and device metadata. This is the foundation for Phase 4 (Memory Manager) but needs the base class now.

```python
class ManagedModel(LazyModel):
    """LazyModel with VRAM tracking for the Memory Manager (Phase 4)."""
    
    def __init__(self, model_id, cache_name, device="cuda",
                 model_class=None, extra_kwargs=None,
                 vram_gb: float = 0, priority: int = 5):
        super().__init__(model_id, cache_name, model_class, device, extra_kwargs)
        self.vram_gb = vram_gb
        self.priority = priority  # 1=hot, 10=cold
        self.last_used = None     # Timestamp for LRU
        self.call_count = 0       # Usage tracking
    
    def get(self):
        import time
        self.last_used = time.time()
        self.call_count += 1
        return super().get()
```

No breaking changes — all existing [LazyModel](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py#19-105) usage continues working. Services can optionally migrate to `ManagedModel`.

##### [NEW] [shared/totalseg_runner.py](file:///d:/new%20manthana%20radiology/manthana-backend/shared/totalseg_runner.py)

**What it does:** Centralized TotalSegmentator wrapper used by 4 services (abdominal_ct, cardiac_ct, brain_mri, spine_neuro). Downloads weights on first call via `totalsegmentator` PyPI package.

```python
"""Manthana — TotalSegmentator Runner (Shared)
Downloads model weights on first run to GPU.
Used by: abdominal_ct, cardiac_ct, brain_mri, spine_neuro.

Supported tasks:
  CT:  "total" (117 structs), "heartchambers", "vertebrae_body", "lung_vessels"
  MRI: "total_mr" (80 structs)
"""

from totalsegmentator.python_api import totalsegmentator

def run_totalseg(input_path, output_dir, task="total", fast=False, device="gpu"):
    totalsegmentator(input_path, output_dir, task=task, fast=fast, device=device)
    # Parse NIfTI outputs → return dict of {structure_name: segmentation_mask}
```

##### [NEW] [shared/comp2comp_runner.py](file:///d:/new%20manthana%20radiology/manthana-backend/shared/comp2comp_runner.py)

**What it does:** Wraps the FDA 510(k) cleared Comp2Comp pipelines (AAQ + BMD). Downloads from GitHub on first run.

```python
"""Manthana — Comp2Comp FDA Runner
FDA 510(k) K243779 (AAQ) + K242295 (BMD)
Downloads weights on first inference from StanfordMIMI/Comp2Comp.
"""

from comp2comp.inference_pipeline import InferencePipeline

def run_aaq(ct_volume_path) -> dict:
    """Returns: {"max_aorta_diameter_mm": float, "aaa_detected": bool}"""
    pipeline = InferencePipeline(model_dir="/models/comp2comp/aaq")
    return pipeline.run(ct_volume_path, task="aaq")

def run_bmd(ct_volume_path) -> dict:
    """Returns: {"bmd_score": float, "low_bmd_flag": bool, "t_score_estimate": float}"""
    pipeline = InferencePipeline(model_dir="/models/comp2comp/bmd")
    return pipeline.run(ct_volume_path, task="bmd")
```

---

#### Service: Abdominal CT (Primary Comp2Comp + TotalSeg target)

##### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/08_abdominal_ct/inference.py)

**What changes:**
- [_run_totalseg()](file:///d:/new%20manthana%20radiology/manthana-backend/services/02_brain_mri/inference.py#52-60) → calls `shared/totalseg_runner.py` with `task="total"` instead of returning hardcoded list
- [_run_comp2comp()](file:///d:/new%20manthana%20radiology/manthana-backend/services/08_abdominal_ct/inference.py#37-40) → calls `shared/comp2comp_runner.py` for AAQ + BMD instead of returning `"pending"`
- [_run_radgpt()](file:///d:/new%20manthana%20radiology/manthana-backend/services/08_abdominal_ct/inference.py#41-46) → wire actual RadGPT inference via LazyModel.get()
- Response `models_used` now says `"Comp2Comp AAQ (FDA K243779)"` and `"Comp2Comp BMD (FDA K242295)"`

##### [MODIFY] [requirements.txt](file:///d:/new%20manthana%20radiology/manthana-backend/services/08_abdominal_ct/requirements.txt)

**Add:** `TotalSegmentator>=2.4.0`, `comp2comp>=1.0.0`, `monai>=1.4.0`

---

#### Service: Cardiac CT

##### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/04_cardiac_ct/inference.py)

**What changes:**
- [_run_totalseg_cardiac()](file:///d:/new%20manthana%20radiology/manthana-backend/services/04_cardiac_ct/inference.py#28-31) → calls `totalseg_runner.run_totalseg(task="heartchambers")` instead of hardcoded list
- Returns real chamber segmentation masks + volume measurements
- Adds `comp2comp_runner.run_aaq()` for aortic root measurement

---

#### Service: Brain MRI

##### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/02_brain_mri/inference.py)

**What changes:**
- [_run_totalseg()](file:///d:/new%20manthana%20radiology/manthana-backend/services/02_brain_mri/inference.py#52-60) → calls `totalseg_runner.run_totalseg(task="total_mr")` for 80 MRI structures
- [_run_prima()](file:///d:/new%20manthana%20radiology/manthana-backend/services/02_brain_mri/inference.py#61-68) → wire actual Prima model inference on brain volume (output: embeddings → classification)

---

#### Service: Spine/Neuro

##### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/10_spine_neuro/inference.py)

**What changes:**
- [_run_totalseg_vertebrae()](file:///d:/new%20manthana%20radiology/manthana-backend/services/10_spine_neuro/inference.py#35-38) → calls `totalseg_runner.run_totalseg(task="vertebrae_body")` for individual vertebra segmentation
- Real vertebral body measurements replace hardcoded labels

---

#### Docker

##### [MODIFY] [docker-compose.yml](file:///d:/new%20manthana%20radiology/manthana-backend/docker-compose.yml)

**What changes:**
- Add `/models` volume mount to ALL services (some may be missing it): `- model_cache:/models`
- Add `model_cache` to top-level volumes definition (persistent across deploys)
- Add `MONAI_DATA_DIRECTORY=/models/monai` env var to services that use MONAI

---

### Verification Plan — Phase 1

1. **Unit test:** Create `tests/test_totalseg_runner.py` → mock `totalsegmentator.python_api.totalsegmentator`, verify it's called with correct `task=` parameter for each service
2. **Unit test:** Create `tests/test_comp2comp_runner.py` → mock `InferencePipeline`, verify AAQ returns diameter and BMD returns flag
3. **Integration test (on GPU):** Deploy to GPU server, call `POST /analyze` with `modality=abdominal_ct` and a real CT DICOM → verify response has real `bmd_score` (not `"pending"`) and real `max_aorta_diameter_mm`
4. **Manual verification:** User deploys to GPU and uploads a CT scan → checks that `models_used` arrays show `"Comp2Comp AAQ (FDA K243779)"` instead of placeholder names

---

## Phase 2: Dual-CXR Ensemble (EVA-X + MedRAX-2)

> Add EVA-X as a second CXR model. Ensemble cross-validation. Wire CheXagent report generation.

### Proposed Changes

---

#### Service: Body X-Ray

##### [NEW] [services/01_body_xray/pipeline_evax.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/01_body_xray/pipeline_evax.py)

**What it does:** EVA-X ViT-S/16 inference for CXR pathology classification. Downloads weights from HuggingFace on first call.

```python
evax_model = ManagedModel(
    model_id="hustvl/EVA-X-ViT-S-16",
    cache_name="evax_vits",
    device="cuda",
    vram_gb=4,
    priority=1,  # High priority — CXR is most common modality
)
```

Returns 18 pathology probabilities (same CheXpert labels as MedRAX-2).

##### [MODIFY] [pipeline_chest.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/01_body_xray/pipeline_chest.py)

**What changes:**
- Import `pipeline_evax.run_evax_classification`
- [run_chest_pipeline()](file:///d:/new%20manthana%20radiology/manthana-backend/services/01_body_xray/pipeline_chest.py#47-103) now runs **both** MedRAX-2 AND EVA-X
- Add `_ensemble_scores()` function: averages scores, computes agreement metric
- Add `ensemble_agreement` field to response dict
- Wire [_run_chexagent()](file:///d:/new%20manthana%20radiology/manthana-backend/services/01_body_xray/pipeline_chest.py#130-140) to actually generate narrative report (remove the placeholder string on line 135)
- CheXagent receives the ensembled scores as context for report generation

##### [MODIFY] [requirements.txt](file:///d:/new%20manthana%20radiology/manthana-backend/services/01_body_xray/requirements.txt)

**Add:** `timm>=1.0.0` (EVA-X uses timm ViT architecture)

---

#### Frontend (Optional Enhancement)

##### [MODIFY] [types.ts](../manthana-radio-frontend/lib/types.ts)

**Add to [AnalysisResponse](../manthana-radio-frontend/lib/types.ts#29-43) interface (line 41):**

```typescript
ensemble_agreement?: number;  // 0-1, how much models agreed
analysis_depth?: "triage" | "deep";  // Phase 3: cascade level
```

No other frontend files need modification — the existing `FindingsPanel` already renders `pathology_scores`, [confidence](file:///d:/new%20manthana%20radiology/manthana-backend/services/13_ecg/inference.py#266-276), and `models_used` from the response. The new fields are optional and gracefully ignored if absent.

---

### Verification Plan — Phase 2

1. **Unit test:** `tests/test_ensemble.py` → given two sets of 18 pathology scores, verify `_ensemble_scores()` correctly averages and computes agreement
2. **Integration test (on GPU):** Upload a chest X-ray → verify `models_used` includes both `"MedRAX-2"` and `"EVA-X"`, verify `ensemble_agreement` field is present and between 0-1
3. **Manual verification:** Compare v3 (single model) vs v4 (ensemble) output on 5 known CXR images → verify ensemble gives more balanced scores

---

## Phase 3: Cascade Inference (Triage → Deep)

> Fast lightweight triage on every scan. Deep analysis only when abnormality detected.

### Proposed Changes

---

#### Gateway

##### [NEW] [gateway/triage.py](file:///d:/new%20manthana%20radiology/manthana-backend/gateway/triage.py)

**What it does:** Runs lightweight triage models directly in the gateway process (no service call needed). Decides if deep analysis is required.

```python
"""Manthana — Triage Layer
Runs in the gateway process. Lightweight models only (<4GB total VRAM).
Returns: {"needs_deep": bool, "triage_scores": dict, "triage_time_ms": int}
"""

TRIAGE_MODELS = {
    "xray":      TorchXRayVisionTriage,   # DenseNet121, ~2GB
    "ecg":       ECGFMTriage,             # ecg-fm embeddings, CPU
    "oral":      EfficientNetTriage,       # EfficientNet-B3, ~2GB
    "ct":        TotalSegFastTriage,       # TotalSeg fast mode, ~4GB
}

ABNORMALITY_THRESHOLD = 0.3  # If any pathology > 0.3, send to deep
```

##### [MODIFY] [gateway/main.py](file:///d:/new%20manthana%20radiology/manthana-backend/gateway/main.py)

**What changes to the `/analyze` endpoint (line 56):**

```python
@app.post("/analyze")
async def analyze(modality, file, patient_id, token_data):
    # ... existing file save code ...
    
    # NEW: Phase 3 — Run triage first
    from triage import run_triage
    triage_result = run_triage(saved_path, modality)
    
    if not triage_result["needs_deep"]:
        # Fast return with triage-level results
        return {
            "job_id": job_id,
            "analysis_depth": "triage",
            "findings": triage_result["findings"],
            "impression": "No significant abnormality detected on initial screening.",
            "processing_time_sec": triage_result["triage_time_ms"] / 1000,
            "models_used": triage_result["models_used"],
            # ... standard fields ...
        }
    
    # Abnormality detected → route to deep analysis (existing flow)
    service_url = route_to_service(modality)
    # ... existing service call code stays identical ...
    result["analysis_depth"] = "deep"
    return result
```

**Key design:** The existing service routing code is **untouched**. Triage is a new layer BEFORE routing. If triage says "normal", we return early. If triage says "abnormal", we fall through to the existing flow. Zero breaking changes.

---

#### Frontend

##### [MODIFY] [useAnalysis.ts](../manthana-radio-frontend/hooks/useAnalysis.ts)

**No changes needed.** The hook already handles whatever [AnalysisResponse](../manthana-radio-frontend/lib/types.ts#29-43) shape the backend sends. The `analysis_depth` field is an optional addition to the type, and the UI already shows all findings regardless.

**Optional:** Add a small visual indicator in the scan stage animation — if `analysis_depth === "triage"`, the animation completes faster (skip heatmap/extracting stages). This is ~5 lines of change in the [scanSingleImage](../manthana-radio-frontend/hooks/useAnalysis.ts#90-201) function but not required.

---

### Verification Plan — Phase 3

1. **Unit test:** `tests/test_triage.py` → given a normal CXR image, verify triage returns `needs_deep: false`. Given an abnormal CXR, verify `needs_deep: true`
2. **Performance test:** Time 100 normal scans with triage vs without → verify triage path completes in <3 seconds vs 15-30 seconds for deep
3. **Manual verification:** Upload a clearly normal chest X-ray → verify response comes back in <3 seconds with `"analysis_depth": "triage"`. Upload an abnormal X-ray → verify full deep analysis runs

---

## Phase 4: Smart GPU Memory Manager

> Models dynamically swap in/out of GPU VRAM using LRU eviction.

### Proposed Changes

---

#### Shared Infrastructure

##### [MODIFY] [model_loader.py](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py)

**Add `ModelMemoryManager` singleton class:**

```python
class ModelMemoryManager:
    """GPU VRAM manager with LRU eviction.
    
    Sits between services and models. When VRAM is full,
    evicts the least-recently-used model to CPU (not deleted —
    stays in CPU RAM for fast reload).
    """
    _instance = None
    
    def __init__(self, gpu_budget_gb=None):
        if gpu_budget_gb is None:
            import torch
            if torch.cuda.is_available():
                gpu_budget_gb = torch.cuda.get_device_properties(0).total_mem / 1e9 * 0.85
            else:
                gpu_budget_gb = 0
        self.gpu_budget = gpu_budget_gb
        self.loaded: dict[str, ManagedModel] = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def acquire(self, model: ManagedModel):
        """Ensure model is on GPU, evicting others if needed."""
        while self._used_vram() + model.vram_gb > self.gpu_budget:
            self._evict_coldest()
        return model.get()
    
    def _evict_coldest(self):
        coldest = min(self.loaded.values(), key=lambda m: m.last_used or 0)
        coldest.unload()  # Moves to CPU, weights stay cached on disk
    
    def _used_vram(self) -> float:
        return sum(m.vram_gb for m in self.loaded.values() if m.is_loaded())
```

**Existing `LazyModel.unload()` already exists (line 96-104)** — it does `del self._model` + `torch.cuda.empty_cache()`. Phase 4 upgrades this to move to CPU instead of deleting:

```python
def offload_to_cpu(self):
    """Move model to CPU RAM (faster to reload than from disk)."""
    if self._model is not None and self.device == "cuda":
        self._model = self._model.cpu()
        import torch
        torch.cuda.empty_cache()
```

**All services continue using `model.get()`.** The Memory Manager operates transparently underneath. Zero service code changes.

---

### Verification Plan — Phase 4

1. **Unit test:** `tests/test_memory_manager.py` → create 3 mock ManagedModels (10GB, 15GB, 20GB) with budget 40GB. Load all 3 → verify the oldest gets evicted when a 4th model is requested
2. **Integration test (on GPU):** Monitor `nvidia-smi` VRAM usage while sending requests to 3 different modalities sequentially → verify VRAM stays within budget and models swap correctly
3. **Manual verification:** SSH into GPU server, run `watch -n1 nvidia-smi`, send CXR request (CXR models load), then send MRI request (MRI loads, if VRAM is tight something evicts), verify no OOM crashes

---

## Phase 5: Cross-Modality Correlation Engine

> Pre-identify clinical patterns across multi-modality results before DeepSeek report generation.

### Proposed Changes

---

#### Report Assembly Service

##### [NEW] [services/report_assembly/correlation_engine.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/report_assembly/correlation_engine.py)

**What it does:** Pattern matching engine that identifies clinically significant correlations across modality results.

```python
"""Manthana — Cross-Modality Correlation Engine

Runs BEFORE DeepSeek unified report generation.
Identifies known clinical correlation patterns.
Feeds pre-identified patterns into the LLM prompt.
"""

CORRELATION_RULES = [
    {
        "name": "Heart Failure Indicators",
        "requires": {
            "xray": {"pleural_effusion": ">0.5"},
            "lab_report": {"BNP": ">400 OR NT-proBNP >900"},
            "cardiac_ct": {"chamber_dilation": ">0.3"},
        },
        "min_match": 2,  # At least 2 of 3 must match
        "clinical_significance": "high",
        "action": "Cardiology referral recommended",
    },
    {
        "name": "Metastatic Disease Pattern",
        "requires": {
            "xray": {"lung_lesion": ">0.5 OR mass": ">0.5"},
            "abdominal_ct": {"liver_lesion": ">0.3"},
            "pathology": {"malignancy_score": ">0.6"},
        },
        "min_match": 2,
        "clinical_significance": "critical",
        "action": "Oncology tumor board review recommended",
    },
    {
        "name": "Osteoporotic Fracture Risk",
        "requires": {
            "abdominal_ct": {"bmd_score": "low_bmd_flag==true"},
            "xray": {"fracture": ">0.3"},
        },
        "min_match": 2,
        "clinical_significance": "warning",
        "action": "Endocrinology referral. DXA scan recommended.",
    },
    # ... 15-20 more clinical correlation rules
]

def find_correlations(individual_results: list[dict]) -> list[dict]:
    """Match multi-modality results against known clinical patterns.
    
    Returns: [{
        "pattern": "Heart Failure Indicators",
        "confidence": 0.87,
        "clinical_significance": "high",
        "matching_modalities": ["xray", "lab_report"],
        "action": "Cardiology referral recommended",
    }]
    """
```

##### [MODIFY] [services/report_assembly/main.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/report_assembly/main.py)

**What changes:**
- Before sending to DeepSeek for unified report, run `find_correlations()` on all individual results
- Append correlated findings to the DeepSeek prompt: `"Pre-identified cross-modality correlations: {correlations}"`
- Include `correlations` array in the response

---

#### Frontend

##### [MODIFY] [types.ts](../manthana-radio-frontend/lib/types.ts)

**Add to [UnifiedAnalysisResult](../manthana-radio-frontend/lib/types.ts#140-158) interface (after line 155):**

```typescript
correlations?: {
  pattern: string;
  confidence: number;
  clinical_significance: "critical" | "high" | "warning" | "info";
  matching_modalities: string[];
  action: string;
}[];
```

##### [NEW] [components/analysis/CorrelationCard.tsx](../manthana-radio-frontend/components/analysis/CorrelationCard.tsx)

**What it does:** Renders a cross-modality correlation finding as a highlighted card in the unified results view. Shows which modalities contributed, the confidence, and recommended clinical action. Uses existing design system (glassmorphism, severity colors).

**~40 lines of code.** Styled to stand out from individual modality findings with a 🔗 link icon and connecting lines showing which modalities contributed.

---

### Verification Plan — Phase 5

1. **Unit test:** `tests/test_correlation_engine.py` → given mock results with pleural effusion from CXR + elevated BNP from lab → verify "Heart Failure Indicators" pattern is matched. Given all-normal results → verify no correlations returned
2. **Integration test:** Send multi-model analysis with CXR + Lab Report that have correlated abnormalities → verify unified report response includes `correlations` array with matched patterns
3. **Manual verification:** Use multi-model upload flow in UI → upload a CXR with pleural effusion + a lab report with elevated BNP → verify a correlation card appears in unified results showing "Heart Failure Indicators" with both modalities highlighted

---

## Phase 6: Model Registry + Hot-Swap

> Version management, canary deployments, instant rollback. Zero-downtime model updates.

### Proposed Changes

---

#### Shared Infrastructure

##### [NEW] [shared/model_registry.py](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_registry.py)

**What it does:** Central registry of all model versions, with A/B testing and rollback support.

```python
"""Manthana — Model Registry
Tracks model versions, enables canary deploys and instant rollback.

Registry is stored in /models/registry.json (persistent volume).
"""

REGISTRY_PATH = "/models/registry.json"

class ModelRegistry:
    def __init__(self):
        self.entries = self._load_registry()
    
    def get_active_model_id(self, model_key: str) -> str:
        """Returns the currently active HuggingFace model ID for a key."""
        entry = self.entries[model_key]
        # Canary: 5% of requests go to canary version
        if entry.get("canary") and random.random() < 0.05:
            return entry["canary"]["model_id"]
        return entry["current"]["model_id"]
    
    def rollback(self, model_key: str):
        """Instantly revert to previous model version."""
        entry = self.entries[model_key]
        entry["current"], entry["previous"] = entry["previous"], entry["current"]
        self._save()
    
    def promote_canary(self, model_key: str):
        """Promote canary to current, current to previous."""
        entry = self.entries[model_key]
        entry["previous"] = entry["current"]
        entry["current"] = entry.pop("canary")
        self._save()
```

Each service's inference code changes from:

```python
# Before (hardcoded model ID):
model = LazyModel(model_id="wanglab/MedRAX2", ...)

# After (registry-managed):
from model_registry import ModelRegistry
registry = ModelRegistry()
model = ManagedModel(
    model_id=registry.get_active_model_id("medrax"),
    ...
)
```

##### [NEW] [gateway/admin.py](file:///d:/new%20manthana%20radiology/manthana-backend/gateway/admin.py)

**What it does:** Admin API routes for model management. Protected by admin JWT.

```python
@admin_router.get("/admin/models")           # List all registered models + versions
@admin_router.post("/admin/models/{key}/canary")     # Set canary version
@admin_router.post("/admin/models/{key}/promote")    # Promote canary → current
@admin_router.post("/admin/models/{key}/rollback")   # Rollback to previous
@admin_router.get("/admin/models/{key}/metrics")     # Accuracy/latency metrics
```

##### [MODIFY] [gateway/main.py](file:///d:/new%20manthana%20radiology/manthana-backend/gateway/main.py)

**What changes:** Add `app.include_router(admin_router, prefix="")` to mount admin endpoints.

---

### Verification Plan — Phase 6

1. **Unit test:** `tests/test_model_registry.py` → set canary, verify 5% of `get_active_model_id()` calls return canary. Call `rollback()`, verify current and previous are swapped. Call `promote_canary()`, verify canary becomes current.
2. **Integration test:** Call `POST /admin/models/medrax/canary` with a new model ID → verify registry.json updates. Call `POST /admin/models/medrax/rollback` → verify it reverts.
3. **Manual verification:** On GPU server, run 20 sequential CXR analyses → check logs to verify ~1 of 20 used the canary model, while 19 used the current model.

---

## Complete New File Inventory

| # | File | Phase | Purpose |
|:-:|------|:-----:|---------|
| 1 | `shared/totalseg_runner.py` | 1 | Centralized TotalSegmentator wrapper |
| 2 | `shared/comp2comp_runner.py` | 1 | FDA 510(k) Comp2Comp AAQ + BMD |
| 3 | `services/01_body_xray/pipeline_evax.py` | 2 | EVA-X ViT inference |
| 4 | `gateway/triage.py` | 3 | Lightweight triage layer |
| 5 | `services/report_assembly/correlation_engine.py` | 5 | Cross-modality pattern matching |
| 6 | `shared/model_registry.py` | 6 | Model version management |
| 7 | `gateway/admin.py` | 6 | Admin API for registry |
| 8 | `components/analysis/CorrelationCard.tsx` | 5 | Frontend correlation display |

## Complete Modified File Inventory

| # | File | Phase | Change Summary |
|:-:|------|:-----:|----------------|
| 1 | [shared/model_loader.py](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py) | 1,4 | Add ManagedModel + ModelMemoryManager |
| 2 | [services/08_abdominal_ct/inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/08_abdominal_ct/inference.py) | 1 | Wire TotalSeg + Comp2Comp |
| 3 | [services/04_cardiac_ct/inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/04_cardiac_ct/inference.py) | 1 | Wire TotalSeg heartchambers |
| 4 | [services/02_brain_mri/inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/02_brain_mri/inference.py) | 1 | Wire TotalSeg MRI + Prima |
| 5 | [services/10_spine_neuro/inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/10_spine_neuro/inference.py) | 1 | Wire TotalSeg vertebrae |
| 6 | [services/01_body_xray/pipeline_chest.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/01_body_xray/pipeline_chest.py) | 2 | Add EVA-X ensemble + wire CheXagent |
| 7 | [gateway/main.py](file:///d:/new%20manthana%20radiology/manthana-backend/gateway/main.py) | 3,6 | Add triage layer + admin router |
| 8 | [services/report_assembly/main.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/report_assembly/main.py) | 5 | Add correlation engine before DeepSeek |
| 9 | [manthana-radio-frontend/lib/types.ts](../manthana-radio-frontend/lib/types.ts) | 2,5 | Add ensemble_agreement + correlations |
| 10 | [docker-compose.yml](file:///d:/new%20manthana%20radiology/manthana-backend/docker-compose.yml) | 1 | Add model_cache volume to all services |

## Frontend Changes Summary

| File | Lines Changed | Phase |
|------|:---:|:---:|
| [lib/types.ts](../manthana-radio-frontend/lib/types.ts) | +5 | 2, 5 |
| `components/analysis/CorrelationCard.tsx` | +40 (new) | 5 |
| [hooks/useAnalysis.ts](../manthana-radio-frontend/hooks/useAnalysis.ts) | +5 (optional) | 3 |
| **Total** | **~50 lines** | |

> [!NOTE]
> **All models continue to download on GPU at runtime.** Nothing is baked into Docker images. The `ManagedModel` class extends [LazyModel](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py#19-105) — same HuggingFace Hub download pattern, same `/models/` volume caching, same first-call-triggers-download behavior. Phase 4's Memory Manager adds VRAM awareness on top without changing the download pattern.
