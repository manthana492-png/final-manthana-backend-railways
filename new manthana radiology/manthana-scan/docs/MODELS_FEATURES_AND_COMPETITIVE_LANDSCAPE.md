# Manthana Radiologist Copilot — Models, Features & Competitive Landscape

**Document type:** Engineering / product reference derived from the **`manthana-scan` frontend codebase** (Next.js app + `lib/`).  
**Last reviewed:** Code snapshot as of repository state when this file was authored.

**Important limitations**

- This repo is **primarily the web client**. **Which neural nets actually run**, **versions**, and **regulatory clearances** are enforced by your **gateway / backend services**, not by these UI strings.
- **Model names** in `lib/constants.ts` are **labels for routing and UX** (and gateway ports). They are **not** proof that each named model is deployed, licensed, or FDA/CE/CDSCO-cleared **in your product**.
- **No FDA/CDSCO/CE status** is stored or validated in this frontend for Manthana or for individual models.

---

## 1. Deep dive: models referenced in code (by modality)

Source: `lib/constants.ts` → `MODALITIES[].models`.

| Modality ID   | UI label     | Models named in client (strings) |
|---------------|-------------|-----------------------------------|
| `auto`        | Auto-Detect | `All` (routing placeholder) |
| `xray`        | X-Ray       | **MedRAX-2**, **CheXagent**, **YOLOv8**, **TotalSeg** |
| `ct`          | CT Scan     | **TotalSeg**, **Prima**, **nnUNet**, **MedSAM2**, **RadGPT**, **Comp2Comp** |
| `mri`         | MRI         | **Prima**, **TotalSeg MRI**, **TotalSeg Vertebrae** |
| `ultrasound`  | Ultrasound  | **OpenUS**, **MedSAM2** |
| `ecg`         | ECG         | **ecg-fm**, **HeartLang** |
| `pathology`   | Pathology   | **Virchow** |
| `mammography` | Mammography | **Mirai** (four-view exam for risk scores) |
| `cytology`    | Cytology    | **Virchow Cell** |
| `oral_cancer` | Oral Cancer | **EfficientNet-B3** |
| `lab_report`  | Lab Reports | **DeepSeek-V3** (report / text analysis) |

**API contract:** `lib/types.ts` → `AnalysisResponse.models_used: string[]` expects the **server** to echo which models ran. The UI can display that when the gateway returns it.

---

## 2. Regulatory reality (expert note)

| Topic | What the codebase shows |
|------|-------------------------|
| **Per-model FDA flags** | **None** — no fields like `fda_cleared: boolean` per model. |
| **Product labeling** | `DISCLAIMER` in `lib/constants.ts` states decision-support only; not a diagnostic device. |
| **Third-party names** | Strings like **Mirai**, **TotalSeg**, **nnUNet** refer to **known public models / families**. Some **vendors** publish FDA clearances for **their** productized versions; **your deployment** must match **your** regulatory filing. |

**Do not** market a model as “FDA approved” in **your** product unless **your** SaMD filing / vendor license says so.

---

## 3. Deep dive: clinical / imaging features in the frontend

### 3.1 DICOM & viewer

| Feature | Implementation |
|--------|----------------|
| **DICOM stack viewing** | `components/scanner/DicomViewport.tsx` — **Cornerstone3D**, `lib/cornerstoneInit.ts` |
| **Multi-slice sort** | `InstanceNumber` ordering (`lib/cornerstoneInit.ts`) |
| **Window / level** | `DicomToolbar`, `lib/windowingPresets.ts` (radiologist-style presets) |
| **Metadata overlay** | `DicomMetadata.tsx`, `DicomViewportState` in `lib/types.ts` |
| **MPR** | Commented in code as **future / placeholder** — not a full enterprise MPR workstation |

### 3.2 Heatmaps & attention

| Feature | Implementation |
|--------|----------------|
| **Heatmap URL** | `AnalysisResponse.heatmap_url`, per-finding `Finding.heatmap_url` |
| **Overlay** | `HeatmapOverlay.tsx`, `ScanViewport` ties to `heatmapState` (opacity, scheme, active finding) |
| **Controls** | `HeatmapControls.tsx`, `HeatmapLegend.tsx` |
| **Scan narrative stage** | `SCAN_PHASES` includes `"GENERATING ATTENTION MAP…"` (`lib/constants.ts`) |

### 3.3 RADS / structured reporting (UI scoring)

| Feature | Implementation |
|--------|----------------|
| **Standards library** | `lib/structured-reports.ts` — ACR-style tables: **BI-RADS**, **Lung-RADS**, **TI-RADS**, **PI-RADS**, **LI-RADS** (defined), **Fleischner**-style (CXR), **Bethesda** (thyroid + cervical), **Minnesota Code** (ECG), **TNM/OED** oral, lab reference bands |
| **Registry used for scoring** | `RADS_REGISTRY` maps modalities → one primary standard each (e.g. `ct` → **Lung-RADS**). **Note:** **LI-RADS** is **defined** in the file but **not** currently in `RADS_REGISTRY` (not active in the scoring path). |
| **Scoring engine** | `scoreFindings()` — maps AI **severity + confidence** heuristically to a **category**; this is **UX assistance**, not a certified calculator for every RADS nuance. |
| **UI** | `IntelligencePanel.tsx` — **RADS CLASSIFICATION** badge when `radsScore` is non-null |

### 3.4 AI workflow & integration

| Feature | Implementation |
|--------|----------------|
| **Gateway API** | `lib/api.ts` — `analyze`, job status, `report`, **`copilot`**, **`unified-report`**, PACS routes |
| **Multi-modality** | `useMultiModelAnalysis`, `MultiModelSelector`, upload wizard, unified report request |
| **PACS sidecar** | Orthanc-oriented **studies / worklist / settings** via gateway (`PacsBrowser`, `WorklistPanel`, `PacsSettings`) — **not** a replacement enterprise PACS |

---

## 4. Expert comparison matrix — Manthana (this client) vs named competitors

**Legend**

- **Strong** = category leader / typical enterprise strength.  
- **Medium** = present or partial.  
- **Focus** = different product lane (not worse—different job).

### 4.1 India — named players

| Dimension | **Manthana** (this repo) | **Qure.ai** (e.g. qXR) | **5C Network** |
|-----------|--------------------------|------------------------|----------------|
| **Primary lane** | Multi-modality **AI copilot** + gateway UI | **CXR** screening & programmes; scale | **Teleradiology network** + AI + human QA |
| **Model surface in UI** | Many **named** models across modalities (strings) | **Focused** CXR/TB pipeline depth | **Operational** scale (sites, TAT, radiologists) |
| **DICOM viewer depth** | **Cornerstone3D** stack + tools; not full PACS | Viewer integration per deployment | End-to-end **report delivery** focus |
| **Heatmap / attention** | **Yes** (client hooks + overlay) | Product-specific (e.g. overlays in programmes) | Varies; not comparable 1:1 |
| **RADS-style UI** | **Yes** (structured tables + badge) | Where product scope matches (e.g. TB / findings) | Radiologist report is human-led |
| **FDA/CDSCO story** | **Not in frontend code** — must come from **your** backend | **Vendor-documented** clearances for **their** products | Service + platform compliance |
| **Win angle** | **Breadth + copilot + unified multi-mod** UX | **Depth + evidence** on CXR / public health | **Speed + capacity** + QA network |

### 4.2 Global — named players

| Dimension | **Manthana** (this repo) | **Aidoc** (aiOS / radiology AI) | **Siemens AI-Rad Companion** |
|-----------|--------------------------|----------------------------------|------------------------------|
| **Enterprise orchestration** | **Gateway** + health dots; lightweight | **aiOS** — triage, many algorithms, governance narrative | **OEM** — deep PACS / SR integration |
| **In-PACS DICOM SR** | **Not shown** in client as SR writer | Strong **workflow** integration story | **DICOM SR** into PACS (vendor materials) |
| **Regulatory breadth** | **Your** filing | Many **FDA-cleared** modules (vendor) | **FDA-cleared** modules (vendor) |
| **Copilot / LLM report** | **`/copilot`**, **`/unified-report`**, DeepSeek-named lab path | Product-specific | Less “LLM copilot” in public positioning |
| **Win angle** | **Fast iteration**, **multi-modal synthesis** UX | **Hospital-grade** triage + integration | **Scanner + PACS** ecosystem lock-in |

### 4.3 Adjacent India AI diagnostics (different lane)

| Player | Why compare carefully |
|--------|------------------------|
| **Niramai** | Breast **thermal** screening — different modality than general radiology copilot. |
| **SigTuple** | **Pathology** hardware + cloud — overlaps “AI diagnostics” but not radiology viewer parity. |

---

## 5. Summary — honest positioning

1. **Strength:** The client implements a **credible AI-first radiology copilot shell**: gateway APIs, **DICOM** viewing, **heatmaps**, **RADS-flavored** structured presentation, **multi-model** flows, **PACS** browser — with **broad modality/model naming** in UI config.  
2. **Gap vs OEM / Aidoc-class:** **No** in-repo proof of **enterprise PACS SR**, **hanging protocols**, **audit trails**, or **per-model regulatory** fields — those must be **backend + compliance** if claimed.  
3. **Competitors win** on **single-disease depth + trials** (Qure), **ops + humans** (5C), **integration + clearances** (Aidoc, Siemens). **Manthana can win** on **unified copilot UX and multi-modality orchestration** — if the **gateway** and **clinical validation** match the UI story.

---

## 6. References (files in this repo)

| File | Role |
|------|------|
| `lib/constants.ts` | Modality → **model name strings**, ports, disclaimer |
| `lib/types.ts` | `AnalysisResponse`, findings, heatmap URLs, DICOM state |
| `lib/api.ts` | Gateway endpoints |
| `lib/structured-reports.ts` | RADS definitions + `scoreFindings` |
| `components/scanner/DicomViewport.tsx` | DICOM rendering |
| `components/scanner/Heatmap*.tsx` | Heatmap UX |
| `components/findings/IntelligencePanel.tsx` | RADS badge, findings, heatmap toggles |

---

*This document is for internal engineering and product alignment. It is not legal or regulatory advice.*
