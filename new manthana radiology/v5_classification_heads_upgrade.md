# V5 — Classification Heads & Model Intelligence Upgrade

**All 13 Modalities · All Models Retained · Model Names Hidden · Indian Doctor Market**

---

## Strategy

Manthana serves **individual Indian doctors** — allopathy, Ayurveda, Unani, Siddha, Homeopathy — who need radiologist-grade second opinions. Not Apollo-tier hospital chains. These doctors upload X-ray films, ECG strips, CT films, oral cavity photos from their phone cameras. They need clear, actionable reports they can trust.

**Core principles:**
- **Broad coverage is our moat** — all 13 modalities stay
- **All models run on GPU behind our backend** — no model names cross to frontend
- **Launch fast** with standard quality, deepen each modality post-launch
- **Every model that gives real predictions is better than one that says "pending"**

---

## Phase 0 — Model Name Obfuscation (P0, 1 day)

> No model name should ever cross from GPU backend to any frontend-facing surface.

### The Problem Now

`models_used` arrays currently contain real model names like `"MedRAX-2"`, `"CheXagent-8b"`, `"TotalSegmentator"`, `"DeepSeek-V3"`, `"Virchow"`. These appear in:
- API JSON responses → visible to anyone inspecting network traffic
- [IntelligencePanel.tsx](../manthana-radio-frontend/components/findings/IntelligencePanel.tsx) → renders model names as badges
- [UnifiedReportPanel.tsx](../manthana-radio-frontend/components/findings/UnifiedReportPanel.tsx) → shows model names in unified report

### The Fix

**Backend:** Every service's response `models_used` gets translated through an obfuscation map before leaving the gateway.

#### [MODIFY] [gateway/main.py](file:///d:/new%20manthana%20radiology/manthana-backend/gateway/main.py)

Add a response transform function:

```python
# Model name obfuscation — no real model names leave the backend
MODEL_DISPLAY_NAMES = {
    "MedRAX-2": "Manthana CXR Engine",
    "EVA-X": "Manthana CXR Engine v2",
    "CheXagent-8b": "Manthana Report AI",
    "TorchXRayVision-DenseNet121-triage": "Manthana Triage Engine",
    "ecg-fm": "Manthana ECG Engine",
    "HeartLang": "Manthana ECG Language AI",
    "Prima": "Manthana Neuro Engine",
    "TotalSegmentator-v2": "Manthana Segment Engine",
    "TotalSegmentator-MRI": "Manthana Segment Engine",
    "Virchow (Apache 2.0)": "Manthana Pathology Engine",
    "RadGPT": "Manthana CT Intelligence",
    "Comp2Comp AAQ (FDA K243779)": "Manthana Vascular Analysis (FDA-ref)",
    "Comp2Comp BMD (FDA K242295)": "Manthana Bone Density (FDA-ref)",
    "EfficientNet-B3": "Manthana Oral Screening Engine",
    "DeepSeek-V3": "Manthana Report AI",
    "Mirai": "Manthana Mammography Engine",
    "triage-heuristic": "Manthana Quick Screen",
}

def _obfuscate_model_names(models: list) -> list:
    """Replace real model names with Manthana-branded names."""
    return [MODEL_DISPLAY_NAMES.get(m, "Manthana AI Engine") for m in (models or [])]
```

Apply it on **every response** before returning from `/analyze` and `/unified-report`:

```python
result["models_used"] = _obfuscate_model_names(result.get("models_used", []))
```

> [!IMPORTANT]
> Keep the FDA reference numbers visible (K243779, K242295) — they're publicly documented clearance numbers, not model identifiers, and showing them adds credibility.

#### [MODIFY] [useAnalysis.ts](../manthana-radio-frontend/hooks/useAnalysis.ts)

Remove hardcoded model names from fallback/demo data:

```typescript
// BEFORE:
models_used: ["MedRAX-2", "CheXagent-8b", "TotalSegmentator"],

// AFTER:
models_used: ["Manthana CXR Engine", "Manthana Report AI"],
```

#### [MODIFY] [useMultiModelAnalysis.ts](../manthana-radio-frontend/hooks/useMultiModelAnalysis.ts)

```typescript
// BEFORE:
models_used: ["DeepSeek-V3", ...results.flatMap((r) => r.result.models_used)],

// AFTER:
models_used: ["Manthana Report AI", ...results.flatMap((r) => r.result.models_used)],
```

**Result:** A doctor inspecting network traffic sees `"Manthana CXR Engine"`, not `"MedRAX-2"`. Frontend shows branded engine names. Reverse engineering reveals nothing about the underlying models.

---

## Phase 1 — ECG-FM Classification Head (P1, 2 hours) ⚡

### What We Discovered
ecg-fm **already ships `physionet_finetuned.pt`** — a fine-tuned checkpoint with a real classification head for PhysioNet 2021 multi-label ECG diagnoses (AF, LBBB, RBBB, ST changes, LVH). We were loading the pretrained-only model and generating random scores.

### What To Do

#### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/13_ecg/inference.py)

```python
# Load the FINE-TUNED checkpoint (already available at wanglab/ecg-fm)
ecg_fm_model = LazyModel(
    model_id="wanglab/ecg-fm",
    cache_name="ecg_fm",
    device="cpu",
    extra_kwargs={"checkpoint": "physionet_finetuned.pt"},
)

def _run_ecg_fm(signal: np.ndarray) -> tuple:
    model = ecg_fm_model.get()
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    
    # The finetuned model outputs multi-label sigmoid activations
    # for PhysioNet 2021 SNOMED codes → map to our target labels
    probs = torch.sigmoid(output.logits).squeeze().cpu().numpy()
    
    # Map SNOMED codes to readable labels
    rhythm_scores = {
        "sinus_rhythm": float(probs[SNOMED_MAP["SR"]]),
        "atrial_fibrillation": float(probs[SNOMED_MAP["AF"]]),
        "sinus_tachycardia": float(probs[SNOMED_MAP["STach"]]),
        "sinus_bradycardia": float(probs[SNOMED_MAP["SBrad"]]),
        "st_elevation": float(probs[SNOMED_MAP["STE"]]),
        "st_depression": float(probs[SNOMED_MAP["STD"]]),
        "lvh": float(probs[SNOMED_MAP["LVH"]]),
        "lbbb": float(probs[SNOMED_MAP["LBBB"]]),
        "rbbb": float(probs[SNOMED_MAP["RBBB"]]),
    }
    return features, rhythm_scores
```

**Impact:** ECG goes from `null` scores → real multi-label rhythm classification. An Ayurveda doctor uploading a patient's ECG strip gets actual AF/LVH detection. This is the single highest-ROI fix in the entire plan.

---

## Phase 2 — HeartLang Reclassification (P2, 4 hours)

### What We Discovered
HeartLang is **NOT a report generator**. It's:
- QRS-Tokenizer → heartbeat tokens ("ECG words")
- VQ-HBR → codebook of 8192 heartbeat morphologies
- HeartLang Transformer → masked ECG language model
- Output: **classification labels** (not free text)

Our code assumed it generates clinical narrative text — it doesn't.

### What To Do

#### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/13_ecg/inference.py)

Restructure the ECG pipeline:
1. **ecg-fm** (with finetuned head) → rhythm classification scores
2. **HeartLang** → second classification model (ensemble with ecg-fm for robustness)
3. **DeepSeek** (via report assembly) → narrative report text from scores

```python
# HeartLang as ENSEMBLE classifier, not report generator
def _run_heartlang(signal) -> dict:
    """HeartLang: classification-only (NOT text generation)."""
    model = heartlang_model.get()
    tokens = qrs_tokenizer(signal)  # Tokenize ECG into heartbeat words
    embeddings = model.encode(tokens)
    
    # Linear head on HeartLang embeddings for rhythm classification
    # (fine-tune with --trainable linear on PTB-XL)
    scores = linear_head(embeddings)
    return scores  # Dict of rhythm labels → probabilities

# Ensemble: average ecg-fm + HeartLang scores
ecg_scores = _run_ecg_fm(signal)
hl_scores = _run_heartlang(signal)
final_scores = _ensemble_ecg(ecg_scores, hl_scores)

# Narrative text comes from DeepSeek via report assembly
# NOT from HeartLang
```

---

## Phase 3 — Prima MLP Heads (P1, 1 day)

### What We Discovered
Prima's GitHub repo includes **MLP head architectures and training recipes** for 52 radiologic diagnoses. The paper explicitly describes freezing the volume/sequence/study transformers and training shallow MLPs. License: MIT.

### What To Do

#### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/02_brain_mri/inference.py)

```python
def _run_prima(volume) -> dict:
    model = prima_model.get()
    
    # Step 1: Extract study-level embeddings (frozen encoder)
    embeddings = model.encode_study(volume)
    
    # Step 2: MLP head for clinical classification
    # Architecture from Prima repo: embeddings → 256 → num_classes
    head = prima_classification_head.get()
    logits = head(embeddings)
    probs = torch.sigmoid(logits).cpu().numpy()
    
    return {"scores": {
        "normal": float(probs[0]),
        "mass_lesion": float(probs[1]),
        "hemorrhage": float(probs[2]),
        "infarct": float(probs[3]),
    }}
```

**To get the MLP weights:**
1. Check Prima repo's `downstream/` folder for pre-trained MLP checkpoints
2. If not available: use their training recipe with BraTS dataset (MIT)
3. Architecture: `nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 4))`

---

## Phase 4 — Pathology MIL Aggregation (P2, 2-3 days)

### What We Discovered
- Our backbone (Virchow) produces embeddings — no classification head included
- **DSMIL** (MIT license) is the best commercially-safe MIL aggregator
- AUCs up to 0.96 on Camelyon16, 0.98 on TCGA-Lung
- We add DSMIL as a slide-level aggregation head on top of our existing tile embeddings

### What To Do

#### [MODIFY] [inference.py](file:///d:/new%20manthana%20radiology/manthana-backend/services/05_pathology/inference.py)

```python
# Keep existing backbone for tile embeddings
# Add DSMIL (MIT) for slide-level classification

from dsmil_aggregator import DSMILClassifier

# DSMIL head: bag of tile embeddings → slide-level prediction
dsmil_head = DSMILClassifier(
    input_dim=768,    # Backbone embedding dimension
    num_classes=4,     # benign, malignant, inflammation, necrosis
)

def _aggregate_embeddings(embeddings) -> dict:
    if not embeddings:
        return {"classification_status": "no_embeddings", ...}
    
    import torch
    bag = torch.stack([torch.from_numpy(e).squeeze() for e in embeddings])
    
    # DSMIL dual-stream: max-pool critical instance + attention
    instance_scores, bag_score = dsmil_head(bag)
    probs = torch.sigmoid(bag_score).cpu().numpy()
    
    return {
        "tissue_type": CLASS_NAMES[probs.argmax()],
        "malignancy_score": float(probs[1]),
        "inflammation_score": float(probs[2]),
        "necrosis_score": float(probs[3]),
    }
```

#### [NEW] [shared/dsmil_aggregator.py](file:///d:/new%20manthana%20radiology/manthana-backend/shared/dsmil_aggregator.py)

Wrapper around DSMIL (from `binli123/dsmil-wsi`, MIT). Loads pre-computed attention weights or trains from scratch on first labeled WSI batch.

---

## Phase 5 — Oral Cancer Fine-Tuning Plan (P3, 3-5 days)

### What We Discovered
No pre-trained Normal/OPMD/OSCC model exists anywhere. Must fine-tune our own.

### Training Plan

| Item | Detail |
|------|--------|
| **Base model** | EfficientNet-B3 (keep current backbone — good for phone camera photos) |
| **Datasets** | Kaggle "Oral Cancer Images" (Apache-2.0) + Rahman 2020 (CC-BY-4.0) + ORCHID (CC-BY-4.0) |
| **Label map** | Normal mucosa → 0, Leukoplakia/OSMF → 1 (OPMD), OSCC → 2 |
| **Min images** | ≥500 per class after augmentation (color jitter, flips, random crops) |
| **Training** | Freeze backbone 5 epochs → unfreeze last 2 stages, focal loss, 20-40 epochs |
| **Compute** | 1× RTX 4090, batch 32, ~2 hours |
| **Target** | ≥85% accuracy, 5-fold cross-validation |
| **Save to** | `/models/oral_cancer_finetuned.pt` → loaded by [LazyModel](file:///d:/new%20manthana%20radiology/manthana-backend/shared/model_loader.py#19-125) at runtime |

### Why This Matters for Indian Market
Oral cancer is **India's #1 cancer** (tobacco chewing). Individual doctors in tier-2/3 cities photograph suspicious oral lesions with their phones every day. A reliable Normal/OPMD/OSCC classifier is a killer feature for this market.

---

## Phase 6 — Remaining Classification Head Strategy

For models where training data isn't immediately available, use the **embedding → DeepSeek narrative** pipeline:

```
Model extracts embeddings → runs through whatever head exists → pathology_scores
pathology_scores + structures (text/numbers) → sent to DeepSeek
DeepSeek generates narrative report from that text context
```

This chain **already works** for all 13 modalities. The classification heads in Phases 1-5 upgrade the `pathology_scores` from null/pending to real numbers, which makes DeepSeek's narrative more specific and accurate.

| Modality | Current Status | After V5 |
|----------|:-:|:-:|
| Chest X-Ray | ✅ Real scores (MedRAX-2 + EVA-X ensemble) | ✅ Same |
| ECG | ❌ Null scores | ✅ Real scores (ecg-fm finetuned) |
| Brain MRI | ❌ Null on failure | ✅ Real scores (Prima MLP heads) |
| Pathology | ❌ Null scores | ✅ Real scores (DSMIL aggregation) |
| Oral Cancer | ⚠️ Random head | ✅ Real scores (fine-tuned EfficientNet-B3) |
| Abdominal CT | ✅ Real (Comp2Comp FDA) | ✅ Same |
| Cardiac CT | ✅ TotalSeg + Comp2Comp | ✅ Same |
| Lab Reports | ✅ DeepSeek working | ✅ Same |
| Mammography | ⚠️ Mirai pipeline | ⚠️ Same |
| Ultrasound | ⚠️ Basic pipeline | ⚠️ Same |
| Spine/Neuro | ✅ TotalSeg vertebrae | ✅ Same |
| Cytology | ⚠️ Basic pipeline | ⚠️ Same |

**Post-V5: 8 modalities with real predictions, 5 with basic pipeline + DeepSeek narrative.**

---

## Implementation Order

```
Phase 0 (Day 1)     → Model name obfuscation — SHIP BLOCKER
Phase 1 (Day 1)     → ecg-fm finetuned checkpoint — 2 HOURS
Phase 2 (Day 2)     → HeartLang reclassification — 4 hours
Phase 3 (Day 2-3)   → Prima MLP heads — 1 day
Phase 4 (Day 4-6)   → DSMIL pathology aggregation — 2-3 days
Phase 5 (Day 7-11)  → Oral cancer fine-tuning — 3-5 days
                     ────────────────────────────────
                     Total: ~11 days sequential
```

---

## Key Technical Decisions

### Linear Head vs MLP?
**Start linear** (768 → num_classes). The Perplexity research confirms this is the consensus across ECG-FM, EVA-X, Prima, and HeartLang papers. Move to 2-layer MLP only if linear saturates on your validation set.

### Minimum Training Data per Modality
| Modality | Min Training Set |
|----------|:---:|
| ECG | Already trained (PhysioNet 2021) |
| Brain MRI | ~1-2k labeled studies per class |
| Pathology (DSMIL) | ~200-500 labeled slides per class |
| Oral Cancer | ~500-1000 images per class |
| Chest X-Ray (head already trained) | N/A |
