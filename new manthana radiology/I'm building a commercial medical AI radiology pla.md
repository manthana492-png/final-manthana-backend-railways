<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I'm building a commercial medical AI radiology platform (Manthana) that uses 13 foundation models across different modalities. I have a critical "last mile" problem: foundation models output embeddings, not diagnoses. I need to find pre-trained classification heads, fine-tuned checkpoints, or learn how to train small MLPs to map embeddings → clinical labels for each model.

Models That Need Classification Heads
Please research each of these foundation models and tell me:
Does a pre-trained classification head / fine-tuned checkpoint already exist on HuggingFace, GitHub, or in published papers?
If yes, what is the exact model ID, license, and what clinical labels does it classify?
If no, what is the recommended approach to train one (dataset, training time, compute requirements)?

1. ecg-fm (wanglab/ecg-fm)
Current output: 768-dim embedding vector from ECG signal
Classification I need: Sinus rhythm, Atrial fibrillation, Sinus tachycardia, Sinus bradycardia, ST elevation, ST depression, LVH, LBBB, RBBB
Question: Does ecg-fm ship with any downstream classification heads? Are there any published ECG classification models that take ecg-fm embeddings as input? What about PTB-XL benchmark models that could serve as ready-to-use ECG classifiers (even if not based on ecg-fm)?
2. Virchow (paige-ai/Virchow) — Computational Pathology
Current output: 768-dim tile embeddings from WSI (Whole Slide Image) patches
Classification I need: Malignancy score (benign vs malignant), tissue type classification, inflammation grade, necrosis detection
Question: Does Paige have published classification heads for Virchow? Are there any open-source MIL (Multiple Instance Learning) aggregation heads trained on Virchow embeddings? What's the state of the art for WSI classification from ViT embeddings — CLAM, TransMIL, DSMIL? Which has pre-trained weights on common pathology tasks?
3. Prima (MLNeurosurg/Prima) — Neurosurgery Foundation Model
Current output: Feature vector from brain MRI slices
Classification I need: Normal vs mass lesion, hemorrhage detection, infarct detection, tumor classification (glioma grading)
Question: Does Prima include any downstream task heads? What are the best open-source brain MRI classification models that could replace or complement Prima's embeddings? Look for models trained on BraTS, ADNI, or OASIS datasets.
4. HeartLang (PKUDigitalHealth/HeartLang) — ECG Report Generation
Current output: Natural language ECG interpretation text
Question: Is HeartLang actually a functional model? Does it generate clinical ECG reports from raw signals? What is the input format? Is there a better open-source ECG report generation model?
5. EVA-X (hustvl/EVA-X-ViT-S-16) — Chest X-Ray Foundation Model
Current output: ViT embeddings
I've added: A 18-way linear head initialized with Xavier (random) weights
Question: Does the EVA-X paper or codebase include pre-trained classification heads for CheXpert-14 or NIH-14 labels? Can I download fine-tuned EVA-X weights that already have the classification head trained on CXR pathologies?
Oral Cancer Model Replacement
6. EfficientNet-B3 for Oral Cancer (currently google/efficientnet-b3)
Problem: I'm using ImageNet-pretrained EfficientNet-B3 with num_labels=3 (Normal, OPMD, OSCC). This initializes a RANDOM classification head. Until fine-tuned on actual oral cancer images, predictions are mathematically random.
What I need:
A pre-trained oral cancer / oral lesion classification model on HuggingFace or GitHub
Specifically for: Normal tissue vs OPMD (Oral Potentially Malignant Disorder) vs OSCC (Oral Squamous Cell Carcinoma)
Must be: open-source, commercially usable (Apache 2.0, MIT, BSD), NOT AGPL/GPL
Look for models trained on the ORCA dataset, Oral Cancer (OC) dataset from Kaggle, or any published oral cavity classification model
If no pre-trained model exists: what is the best labeled oral cancer dataset I can use to fine-tune EfficientNet-B3 or a similar model? How many images, what labels, how long to train?
General Research Questions
Linear probing vs MLP heads: For medical foundation models, is a single linear layer sufficient as a classification head, or do I need a 2-3 layer MLP? What's the published consensus for medical imaging tasks?
Training data requirements: For each modality above, approximately how many labeled examples do I need to train a reliable classification head on top of a pre-trained foundation model? Published benchmarks suggest different numbers — I want the minimum viable training set size for each.
FDA-cleared open-source models with classification heads: List ALL open-source AI/ML models (March 2026) that:
Are used in an FDA 510(k) cleared device OR are validated against FDA standards
Have BOTH the foundation model AND classification head freely available
Are commercially usable (Apache 2.0, MIT, BSD licenses)
Cover any of: radiology, pathology, cardiology, dermatology, ophthalmology
Pre-trained MIL aggregation models: For computational pathology (Virchow embeddings → slide-level classification), list the best open-source MIL frameworks with pre-trained aggregation heads:
CLAM (from Mahmood Lab)
TransMIL
DSMIL
HIPT
Which has the best pre-trained weights and easiest integration?
Please provide specific HuggingFace model IDs, GitHub repositories, paper DOIs, and license information for every model or dataset you mention.

Below is a model‑by‑model breakdown, then answers to the general questions and the FDA‑cleared / MIL parts. I’ll call out when something is **commercially safe** (MIT/Apache/BSD) vs **non‑commercial / restrictive**.

***

## 1. ECG‑FM (wanglab/ecg‑fm)

### 1.1 Does ECG‑FM ship classification heads?

- **Repo / weights**
    - GitHub: `bowang-lab/ECG-FM` (MIT).[^1][^2]
    - Weights: HuggingFace `wanglab/ecg-fm` (MIT; weights only).[^3][^4][^5]
- **Checkpoints**
    - `mimic_iv_ecg_physionet_pretrained.pt` – self‑supervised pretrain on MIMIC-IV-ECG + PhysioNet 2021.[^5][^1]
    - `physionet_finetuned.pt` – fine‑tuned from that pretrain on PhysioNet 2021 for **multi‑label ECG diagnosis**.[^2][^1]
    - Training configs for **diagnosis finetuning** are provided (`diagnosis.yaml`), which instantiate a classification head on top of the transformer encoder.[^1]

**So yes:** ECG‑FM provides a downstream **classification head and fine‑tuned checkpoint** (for PhysioNet 2021 style multi‑label arrhythmia / conduction / ST‑T diagnoses). The head is implemented inside fairseq_signals; you don’t get a separate “embeddings‑only → head” artifact, but the full finetuned model is there.

- **Labels**
The PhysioNet 2021 Challenge label set includes common ECG diagnoses like AF, I‑AVB, LBBB, RBBB, ST elevation, ST depression, etc., mapped to SNOMED codes. ECG‑FM’s `physionet_finetuned.pt` is trained on that label set, not exactly your 9‑class schema but overlapping strongly (AF, ST‑changes, BBB, LVH proxies, etc.).[^6][^7][^8][^5]


### 1.2 Are there existing models that take ECG‑FM embeddings as input?

- ECG‑FM’s paper and repo **fine‑tune the full model end‑to‑end** (encoder + head), not a separate MLP on frozen embeddings.[^6][^5][^1]
- No public code/model was found that:
    - Exports fixed 768‑d ECG‑FM embeddings, and
    - Trains a standalone MLP head on those embeddings.

You’ll need to implement this yourself (trivial in PyTorch once you wrap ECG‑FM).

### 1.3 PTB‑XL benchmark ECG classifiers

- **helme/ecg_ptbxl_benchmarking** – canonical PTB‑XL deep‑learning baseline repo (multiple CNN/RNN architectures, good code, pretrained models available) but **GPLv3**, so **not usable in a commercial product**.[^9][^10][^11]
- There are several **MIT‑licensed experimental PTB‑XL classifiers**, e.g.:
    - `NicholasKanos/PTBXL-Dataset-Thesiswork` (multiple CNN/Transformer/LSTM models; MIT).[^12]
    - Others exist but typically share code, not production‑ready weights.

For Manthana, treat these as **reference architectures**, not plug‑and‑play commercial components.

### 1.4 How to train your own ecg‑fm classification head

Given your target labels (SR, AF, ST‑changes, LVH, LBBB/RBBB):

- **Dataset:**
    - Best: large internal ECG corpus with cardiologist labels for your 9 classes.
    - Public starting point: **MIMIC‑IV‑ECG** and/or **PTB‑XL** (form + rhythm labels) mapped into those categories.[^13]
- **Architecture:**
    - Freeze ECG‑FM encoder → 768‑d embedding per recording (usually pooled over time).
    - Start with a **single linear layer** (768 → 9), sigmoid outputs for multi‑label; then consider a 2‑layer MLP (768 → 512 → 9 with GeLU + dropout) if linear underfits. This matches ECG‑FM’s own finetuning style (MLP on frozen features, then optionally unfreeze).[^1][^6]
- **Training recipe (minimum viable):**
    - Data: on the order of **5–10k labeled ECGs** total (hundreds to low thousands per major label) will usually give a decent head; ECG‑FM shows label‑efficiency, but AUROCs stabilize with several thousand labelled examples.[^5][^6]
    - Compute: 1 × A100/RTX 4090, batch 256, 50–100 epochs; ECG‑FM’s own finetuning configs run 140 epochs with batch 256 on a single GPU.[^1]

***

## 2. Virchow / Virchow2 (Paige AI) – Pathology

You’re using Virchow (or Virchow2) as a **tile‑level feature extractor**.

### 2.1 Pre‑trained heads or MIL aggregators from Paige?

- **Virchow (Nat Med “foundation model for clinical‑grade computational pathology”)** is a ViT‑H/14 feature backbone trained on ≈1.5M WSI patches.[^14]
- **Virchow2** on HuggingFace (`paige-ai/Virchow2`) exposes only **feature embeddings** (class token, patch tokens, 2560‑d embedding) and explicitly positions itself as a **“feature backbone”**; no slide‑level classifier weights are included.[^15][^16]
- Publications describe multiple slide‑level tasks (pan‑cancer detection, rare cancer detection, etc.) but **Paige has not released those MIL heads or slide‑level checkpoints.**[^17][^14]

So: no official pre‑trained MIL head specifically for Virchow embeddings is available under a permissive license.

### 2.2 Licensing

- **Virchow2 on HuggingFace** – **CC‑BY‑NC‑ND 4.0**, explicitly non‑commercial, forbids derivatives and using models trained on Virchow outputs in commercial settings.[^16]
- **Virchow in Azure AI model catalog** – shows **Apache‑2.0** license and is deployable as a “frozen feature extractor” inside Azure.[^14]
    - That’s promising for **commercial** usage, but probably constrains you to Azure’s environment and T\&Cs. You cannot just scrape the weights from Azure and redistribute.

For Manthana as a GPU‑hosted SaaS, you’ll almost certainly need a **direct commercial license from Paige/Microsoft** if you want Virchow; otherwise it’s not safe.

### 2.3 State of the art MIL heads / pre‑trained MIL frameworks

All of these are **slide‑level aggregators you can stick on top of Virchow embeddings** (after verifying licenses):

- **DSMIL**
    - Repo: `binli123/dsmil-wsi` (MIT).[^18][^19]
    - Type: dual‑stream MIL (max‑pool critical instance + attention stream).
    - Pre-trained: code ships **pre‑computed features** and training scripts for Camelyon16 and TCGA Lung; you train the head yourself, but it converges quickly on those datasets.[^19][^20]
    - **Commercial use:** MIT → safe.
- **CLAM** (Mahmood Lab)
    - Repo: `mahmoodlab/CLAM` (GPLv3).[^21][^18]
    - Strong performance and good tooling; widely used.
    - **Non‑commercial** due to GPLv3 and, in slideflow-gpl, explicitly noted as GPLv3‑only.[^21]
- **TransMIL**
    - Repo: `szc19990412/TransMIL` (GPLv3, “non‑commercial academic purposes”).[^22][^23]
- **HIPT**
    - Repo: `mahmoodlab/HIPT` (Commons Clause – non‑commercial).[^24]

**For Manthana**, the only clearly **commercial‑safe MIL aggregator with code and examples is DSMIL (MIT)**. You can:

1. Use Virchow embeddings (or any ViT/ResNet features) as DSMIL bag features.
2. Train DSMIL on your WSI labels for:
    - benign vs malignant
    - tissue subtype
    - inflammation grade
    - necrosis presence

For purely open‑source + commercial: **Virchow2 is not usable; DSMIL is.** You’d combine DSMIL with **another backbone** (e.g., a ViT backbone you train on TCGA under Apache/MIT).

***

## 3. Prima (MLNeurosurg/Prima) – Brain MRI

### 3.1 Does Prima include downstream task heads?

- Repo: `MLNeurosurg/Prima` (MIT).[^25][^26][^27]
- Paper: “Learning neuroimaging models from health system‑scale data” (Prima VLM; Nature Medicine‑style VLM).[^26][^27]
- Architecture: hierarchical ViT over study‑level MRI, CLIP‑style training with MRI–report pairs; then **frozen encoder + MLP heads for 52 radiologic diagnoses and several clinical tasks.**[^25][^26]
- **Downstream heads:**
    - The repo includes code for **training the task‑specific MLPs** on the frozen study representation and evaluating them; it also advertises an **“end‑to‑end ready‑to‑use inference pipeline”** for raw MRI to predictions.[^25]
    - The paper explicitly says they freeze volume/sequence/study transformers and train MLPs for radiologic and clinical predictions.[^27][^25]
    - Pre‑trained MLP checkpoints are not clearly surfaced in the top‑level README; you may need to check sub‑folders or contact authors. But you do have **exact architectures and training recipes** under MIT.

So: **yes, Prima defines and trains MLP heads** for multi‑label radiologic diagnosis, but pre‑exported MLP+encoder weights may require some digging or direct contact.

### 3.2 Alternatives / complements – open brain MRI classifiers

If you want explicit classifiers (normal vs mass / hemorrhage / infarct / glioma):

- **fastglioma (MLNeurosurg)** – “visual foundation model enabling rapid, accurate detection of glioma infiltration during surgery”; separate repo under same lab (MIT) focused on tumor infiltration.[^28][^29]
- **Segmentation‑based pipelines** on BraTS (not pure classifiers, but can be turned into lesion presence / grade indicators):
    - nnU‑Net trained on BraTS (multiple open repos; nnU‑Net itself is GPL‑3 so not ideal for commercial core, but can guide architecture).[^30]
- **Research‑grade classifiers on ADNI/OASIS** (Alzheimer’s vs normal, MCI, etc.) tend to be code‑only; licenses vary and are often non‑commercial.

Realistically, **Prima is your best open‑source, commercially usable (MIT) MRI foundation + head solution** today.

***

## 4. HeartLang (PKUDigitalHealth/HeartLang)

### 4.1 Is HeartLang a functional model? What does it do?

- Repo: `PKUDigitalHealth/HeartLang` (MIT).[^31]
- Paper: “Reading Your Heart: Learning ECG Words and Sentences via Pre‑training ECG Language Model” (ICLR 2025).[^32][^33][^34]
- Core idea:
    - **QRS‑Tokenizer** turns ECG signals into a sequence of discrete “heartbeat tokens” (“ECG words” and “sentences”).[^35][^31]
    - **VQ‑HBR**: vector‑quantized heartbeat reconstruction model to learn a large codebook of heartbeat morphologies (8192 codes).[^31]
    - **HeartLang Transformer**: learns masked ECG “language” over heartbeat tokens (self‑supervised).[^33][^35]


### 4.2 Does it generate free‑text clinical reports from raw ECG?

- No. HeartLang is **not a natural‑language report generator**.
- It produces **ECG sentence embeddings / token sequences**, then fine‑tunes classification heads for tasks (PTB‑XL subsets, CPSC2018, Chapman‑Shaoxing) – i.e., it is a **representation + classifier framework**, not ECG→text.[^34][^32][^31]

For ECG report generation, other work like **ECG‑Chat** (ECG‑language LLM) exists, but licenses and robustness are not yet at a “drop this into clinical SaaS” level.[^36]

### 4.3 Input format, checkpoints

- Inputs: EDF/PhysioNet formats → processed into ECG sentences via `QRSTokenizer.py` plus dataset‑specific preprocessing for PTB‑XL / CPSC / CSN.[^31]
- HuggingFace: HeartLang pretraining and VQ‑HBR checkpoints are available on HF (model IDs referenced in the README; HF cards not in the snippets but indicated as “checkpoint is now available on 🤗”).[^32][^31]
- Downstream fine‑tuning: scripts to train **linear probes or small MLP heads** on PTB‑XL form/rhythm, CPSC2018, CSN; examples use a linear trainable head (`--trainable linear`).[^31]

So HeartLang is **a good alternative ECG foundation model + classification framework under MIT**, but **not** a text‑report generator.

***

## 5. EVA‑X (hustvl/EVA‑X, EVA‑X‑ViT‑S‑16)

### 5.1 Do pre‑trained classification heads/checkpoints exist?

- Repo: `hustvl/EVA-X` – code and **pre‑trained backbones** EVA‑X‑Ti / S / B for CXR self‑supervised learning.[^37][^38]
- Paper and docs show EVA‑X is evaluated on **ChestX‑ray14, CheXpert, MIMIC‑CXR**, obtaining SOTA mAUC via **a simple linear classification head on pooled ViT features**.[^39][^40][^41]
- However:
    - The GitHub repo releases **pre‑trained backbones**, plus **fine‑tuning scripts** for classification/segmentation, not ready‑made **CheXpert‑14 or NIH‑14 classifier checkpoints** with heads.[^38][^37]
    - HuggingFace has some third‑party ports (e.g. `MapleF/eva_x`), but they are **backbone models**, not full pathology classifiers.[^42]

So: there is **no official CheXpert‑/NIH‑14‑head checkpoint** you can just download and use. You correctly added your own 18‑way linear head; you now need to train it.

### 5.2 License implications

- EVA‑X paper explicitly notes **license: CC BY‑NC‑SA 4.0**.[^38]
- GitHub repo doesn’t state a more permissive override, and promotional posts say “fully open‑source” but do **not** contradict the non‑commercial license.[^43][^37]

That means **EVA‑X is non‑commercial without a separate license agreement**, regardless of how you implement the head. For Manthana as a commercial platform, you’d need explicit permission / a commercial license from the authors.

### 5.3 Training your EVA‑X head (technical, even if license is solved)

- Dataset: CheXpert or MIMIC‑CXR; EVA‑X was trained on ≈520k unlabeled CXRs from ChestX‑ray14, CheXpert, MIMIC‑CXR.[^40][^39]
- Architecture: start with a **linear head** on global average pooled ViT features (this is what EVA‑X uses to demonstrate SOTA few‑shot performance).[^39][^40]
- Sample complexity:
    - EVA‑X paper shows strong **few‑shot** performance – e.g. COVID‑19 detection at ~95% accuracy with 1% labels – but for robust multi‑label disease classification in production, plan on **at least a few thousand labeled studies per broad label group** (order of 10–20k studies total).[^41][^40]
- Compute: ViT‑S/16 (<25M params) is light: **1 × 16GB GPU, batch 32–64, 10–20 epochs** is often enough for CheXpert‑scale fine‑tuning.

***

## 6. Oral cancer model replacement (EfficientNet‑B3 baseline)

### 6.1 Is there a pre‑trained, 3‑class (Normal / OPMD / OSCC) open model?

From HF/GitHub and the literature:

- Many codes exist (ResNet, NasNet, custom CNNs) for oral cancer/OPMD classification, but:
    - They often target **binary** (normal vs cancer) from smartphone photos (e.g. OCI dataset).[^44][^45]
    - Or histopathology two‑class (normal vs OSCC) classifiers based on datasets like Rahman 2020 (CC BY 4.0).[^46]
    - Or more granular 5‑class tasks in research, but **without released weights** and/or with unclear or non‑commercial licenses.[^47][^48][^49]
- No **HuggingFace or GitHub model with:**
    - `num_labels=3` for **Normal vs OPMD vs OSCC**,
    - pre‑trained weights, and
    - **MIT/Apache/BSD** license
was found.

Conclusion: **you will have to fine‑tune your own classifier**.

### 6.2 Best datasets for fine‑tuning (commercial‑usable)

Given your need for commercial use, prioritize datasets with **Apache‑2.0 or CC‑BY‑4.0**:

1. **Oral Cancer Images for Classification (Kaggle)**
    - “Oral Cancer Images for Classification” dataset, Kaggle; license: **Apache‑2.0**.[^50]
    - Contains oral lesion photographs for classification; check whether labels distinguish OPMD vs OSCC vs normal (some Kaggle variants are just cancer vs non‑cancer).
2. **Histopathological OSCC datasets (normal vs OSCC)**
    - Rahman et al. 2020 “Histopathological imaging database for oral cancer analysis” – released under **CC‑BY 4.0**, 528 images @100× and 696 images @400×, normal vs OSCC.[^46]
    - ORCHID histology dataset (OSMF vs OSCC, with grade‑level labels) – on Zenodo under **CC‑BY 4.0**.[^51]
3. **NDB‑UFES oral cancer \& leukoplakia dataset** – histopathology of OSCC vs leukoplakia (an OPMD); license appears open but you must check exact terms.[^52]
4. **SMART‑OM and related smartphone oral mucosa datasets** – comprehensive oral mucosa images with Normal/Variation/OPMD/OC labels; licensing needs to be checked case by case.[^53]

You can synthesize a 3‑way task by mapping:

- Normal mucosa → Normal
- Histologically confirmed leukoplakia / OSMF / other OPMDs → OPMD
- OSCC labels → OSCC


### 6.3 Training plan for EfficientNet‑B3 (or ViT)

- **Architecture:** EfficientNet‑B3 or ViT‑B/16; both work. EfficientNet has good inductive bias for natural photographic images; ViT can be competitive given enough data.
- **Labelled data requirements (minimum):**
    - For reasonably robust 3‑way classifier, aim for **≥500–1,000 images per class** after augmentation; more is better, especially for OPMD vs OSCC separation where morphology is subtle.
- **Compute:**
    - 1 × 12–16GB GPU, batch 32; 20–40 epochs with augmentations (color jitter, flips, random crops) is typically enough.
- **Training details:**
    - Start by freezing backbone and training only the classification head for a few epochs, then unfreeze last 1–2 stages and fine‑tune with lower LR.
    - Use class‑balanced loss (e.g. focal loss or class‑weighted BCE/CE) if classes are imbalanced (OPMD often under‑represented).

***

## General questions

### A. Linear probes vs 2–3 layer MLP heads

From ECG‑FM, EVA‑X, Prima, HeartLang, and multiple medical VFM fine‑tuning studies:

- **Most foundation‑model papers report very strong performance with a simple linear head** on top of frozen embeddings:
    - EVA‑X: uses **global pooled ViT features + linear layer** for CXR14/CheXpert and shows SOTA mAUC.[^40][^41]
    - ECG‑FM: demonstrates multi‑label ECG interpretation with frozen encoder + linear head (then sometimes unfreezing encoder).[^6][^5]
    - HeartLang: default fine‑tuning setting uses **linear probe** (`--trainable linear`) on top of ECG sentence embeddings.[^31]
    - Prima: uses **shallow MLPs** on frozen study features; essentially linear or 1‑hidden‑layer heads.[^25]
- More complex heads (2–3 layer MLP) help when:
    - You have **moderate labelled data** (thousands to tens of thousands) and need some non‑linearity, *and*
    - You keep the backbone frozen to constrain compute / stability.
    - CUFIT (Curriculum Fine‑tuning for Vision Foundation Models) explicitly combines **linear probes + adapters** to handle noisy medical labels, but still keeps classifier simple.[^54]
- **Consensus:**
    - **Start with a linear head.** It’s standard, easy to debug, and gives you a clean sense of embedding quality.[^55][^56]
    - Move to a **small 2‑layer MLP** only if linear saturates and you have enough labels; over‑parameterized heads can overfit small medical datasets.


### B. Training data requirements per modality (minimum viable)

Approximate “MVPS” (minimum viable performance scale) from the cited works:

- **ECG (ECG‑FM, HeartLang, ECGFounder, etc.)**
    - ECG‑FM: shows strong label‑efficiency but still scales performance with more labeled ECGs; AUROC for AF and LVEF<=40% remains high with a few thousand labeled examples.[^5][^6]
    - For a 9‑label arrhythmia/ST‑change head, target **≥5k–10k ECGs** total (hundreds+ per class, thousands for common rhythms/AF).
- **Chest X‑ray (EVA‑X, TorchXRayVision)**
    - EVA‑X: 95% COVID‑19 accuracy using **1% labels** (few thousand images), but robust multi‑label CheXpert classifiers in practice tend to train on **10–50k studies**.[^41][^40]
    - For your platform: plan on **≥10k CXRs** with labels for stable 10–20‑way head; more if you want robust calibration.
- **Brain MRI (Prima)**
    - Prima’s prospective evaluation covered ~30k MRI studies across **52 diagnoses**; but that’s the foundation pretraining stage.[^26][^27]
    - For a smaller, focused head (e.g., normal vs mass / hemorrhage / infarct / glioma), **1–2k labeled studies per major category** is a good target.
- **Pathology WSI (Virchow‑like + DSMIL)**
    - CLAM/DSMIL/HIPT style MIL methods typically show strong performance with **200–500 slides per class** on Camelyon16 / TCGA.[^57][^58][^20][^19]
    - For benign vs malignant, necrosis detection, etc., try for **a few hundred slides per endpoint**, more if label noise is high.
- **Oral lesion photographs**
    - Recent smartphone‑based oral datasets (<3k images) already get ~90–95% accuracy in 2–5 class tasks with strong augmentations.[^47][^44]
    - For your 3‑way Normal/OPMD/OSCC: **≥500–1,000 images per class** is a realistic minimal goal.

These are **orders of magnitude**, not hard thresholds; the foundation models do buy you label‑efficiency, but you still need hundreds–thousands of clean labels by task.

### C. FDA‑cleared open‑source models with classification heads

Re‑checking against FDA listings + open‑source code:

- **Comp2Comp AAQ and BMD (Bunkerhill AAQ \& BMD)**
    - Two CT pipelines (AAA diameter, opportunistic spinal BMD) that:
        - Are used **as is** inside 510(k)‑cleared devices Bunkerhill AAQ (K243779) and Bunkerhill BMD (K242295).[^59][^60][^61][^62]
        - Have fully open‑sourced code and model weights in `StanfordMIMI/Comp2Comp`.[^63][^64][^65]
        - Include end‑to‑end pipelines: CT DICOM → segmentation (nnU‑Net / TotalSegmentator) → **MLP/logistic classification or regression head** (AAA diameter / BMD flag), all in the repo.[^65][^66][^63]
        - Are described as **Apache‑2.0** in org metadata and paper (but check the GPLv3 mention in older docs with counsel).[^67][^68][^69]

No other FDA‑cleared AI/ML imaging device could be found where **both the backbone and the clinical head (with weights) are open‑sourced under a permissive commercial license**. Everything else either:

- keeps the classifier proprietary (CT Cardiomegaly, LV function tools, etc.), or
- uses open frameworks (MONAI, PyTorch, TotalSegmentator) but not open weights for the diagnostic head.[^70][^71][^72][^73]

So your “FDA‑cleared + open model + open head + commercial license” set is effectively **Comp2Comp AAQ \& BMD only** at present.

***

## D. Best pre‑trained MIL aggregation options for Virchow‑like features

Given your need for an MIL head over Virchow embeddings:

- **DSMIL (binli123/dsmil‑wsi)** – **Recommended**
    - License: **MIT** → compatible with Manthana.[^19]
    - Pre-trained artifacts: precomputed feature vectors and configs for Camelyon16 \& TCGA Lung, with reported AUCs up to 0.96 on Camelyon16, 0.98 on TCGA‑Lung.[^20][^19]
    - Integration: treat your Virchow 2560‑d tile embeddings as DSMIL “instances” and train slide‑level classifiers for malignancy, tissue type, inflammation, necrosis.
- **CLAM (mahmoodlab/CLAM)**
    - Strong data‑efficient MIL baseline (Nature BME) but **GPLv3** and re‑distributed in `slideflow-gpl` as GPL‑only.[^18][^57][^21]
    - Pre-trained configs \& sometimes weights exist for Camelyon16, TCGA cohorts, but cannot be used commercially.
- **TransMIL, HIPT, CAMIL, TDT‑MIL**
    - Provide state‑of‑the‑art research code, often with pretraining on TCGA/Camelyon16, but are licensed under **GPLv3, Commons Clause, or no‑clear‑commercial license**.[^58][^74][^75][^22][^24]

For a **commercial Virchow‑like MIL stack** today, the safest strategy is:

1. Use a **permissively licensed backbone** (e.g. a ViT or ResNet trained on TCGA under Apache/MIT, or a model like MedImageInsight if its MIT implementation suits your needs).[^76]
2. Use **DSMIL** as your MIL aggregator.
3. If you secure a **commercial Virchow license**, drop its tile embeddings into the same DSMIL head.

***

If you’d like, I can next map this into a concrete **head training plan per model** for Manthana (exact PyTorch heads, loss functions, and minimal dataset sizes per task), or help you prioritize which foundation models to keep vs replace given license and FDA ambitions.
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^120][^121][^122][^123][^124][^125][^126][^127][^128][^129][^130][^131][^132][^133][^134][^135][^136][^137][^138][^139][^140][^141][^142][^143][^144][^145][^146][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://github.com/bowang-lab/ecg-fm

[^2]: https://huggingface.co/api/resolve-cache/spaces/mystic-cbk/ecg-fm-api/332d02978908d3bbda5f4319135bfe0acdbf9cc6/ecg_fm_github_readme.md?download=true\&etag="8d9cbb31d9bc56e301be3465e146963063430afd"

[^3]: https://huggingface.co/wanglab/ecg-fm

[^4]: https://huggingface.co/wanglab/ecg-fm/raw/main/README.md

[^5]: https://arxiv.org/abs/2408.05178

[^6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324/

[^7]: https://physionet.org/content/challenge-2021/

[^8]: https://www.semanticscholar.org/paper/839f377860e29a8c3b184053613f20e7d9e0ca56

[^9]: https://github.com/helme/ecg_ptbxl_benchmarking/

[^10]: https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/LICENSE

[^11]: https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/README.md

[^12]: https://github.com/NicholasKanos/PTBXL-Dataset-Thesiswork

[^13]: https://physionet.org/content/ptb-xl/1.0.3/

[^14]: https://ai.azure.com/catalog/models/Virchow

[^15]: https://huggingface.co/paige-ai/Virchow2

[^16]: https://www.aimodels.fyi/models/huggingFace/virchow2-paige-ai

[^17]: https://cbirt.net/the-power-of-foundation-models-in-pathology-how-paiges-virchow-is-changing-cancer-detection/

[^18]: https://github.com/orgs/mahmoodlab/repositories

[^19]: https://github.com/binli123/dsmil-wsi

[^20]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8765709/

[^21]: https://pypi.org/project/slideflow-gpl/

[^22]: https://liner.com/review/transmil-transformer-based-correlated-multiple-instance-learning-for-whole-slide

[^23]: https://github.com/xiongxuechun/TransMIL

[^24]: https://github.com/computationalpathologygroup/hvit

[^25]: https://github.com/MLNeurosurg/Prima

[^26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11838732/

[^27]: https://arxiv.org/html/2509.18638v1

[^28]: https://github.com/MLNeurosurg

[^29]: https://github.com/MLNeurosurg/fastglioma

[^30]: https://github.com/AIM-Harvard/TotalSegmentator-to-nnUNet-format-convert

[^31]: https://github.com/PKUDigitalHealth/HeartLang

[^32]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/17aa70697d6cd35835f201c6fb0a2fd5-Abstract-Conference.html

[^33]: https://arxiv.org/pdf/2502.10707.pdf

[^34]: https://iclr.cc/media/iclr-2025/Slides/30897.pdf

[^35]: https://www.themoonlight.io/en/review/reading-your-heart-learning-ecg-words-and-sentences-via-pre-training-ecg-language-model

[^36]: https://arxiv.org/html/2408.08849v1

[^37]: https://deeplearn.org/arxiv/484940/eva-x:-a-foundation-model-for-general-chest-x-ray-analysis-with-self-supervised-learning

[^38]: https://arxiv.org/html/2405.05237

[^39]: https://ui.adsabs.harvard.edu/abs/2024arXiv240505237Y/abstract

[^40]: https://www.themoonlight.io/zh/review/eva-x-a-foundation-model-for-general-chest-x-ray-analysis-with-self-supervised-learning

[^41]: https://ownyourai.com/eva-x-a-foundation-model-for-general-chest-x-ray-analysis-with-self-supervised-learning/

[^42]: https://huggingface.co/MapleF/eva_x

[^43]: https://x.com/BoWang87/status/1991553217457443047

[^44]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12227544/

[^45]: https://arxiv.org/html/2510.01547v1

[^46]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6994517/

[^47]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12836996/

[^48]: https://scholarshare.temple.edu/bitstream/handle/20.500.12613/10255/Solanki_temple_0225M_15696.pdf?sequence=1\&isAllowed=y

[^49]: https://onlinelibrary.wiley.com/doi/10.1155/2022/7643967

[^50]: https://www.kaggle.com/datasets/muhammadatef/oral-cancer-images-for-classification/data

[^51]: https://ibdc.dbtindia.gov.in/ibia/study_details_browse_l/HISTOS_1000000013/

[^52]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10139872/

[^53]: https://www.nature.com/articles/s41597-026-06954-5

[^54]: https://papers.nips.cc/paper_files/paper/2024/file/2093ed77c549eda95bd6f7212b735b43-Paper-Conference.pdf

[^55]: https://arxiv.org/html/2603.16840v1

[^56]: https://www.sei.cmu.edu/documents/6343/25_Towards_Better_Understandin.pdf

[^57]: http://arxiv.org/pdf/2004.09666.pdf

[^58]: https://arxiv.org/html/2405.15127v1

[^59]: https://www.accessdata.fda.gov/cdrh_docs/pdf24/K243779.pdf

[^60]: https://www.bunkerhillhealth.com/resources/bunkerhill-bmd-clearance

[^61]: https://healthimaging.com/topics/medical-imaging/diagnostic-screening/opportunistic-screening-tool-osteoporosis-nabs-fda-clearance

[^62]: https://www.accessdata.fda.gov/cdrh_docs/pdf24/K242295.pdf

[^63]: https://www.arxiv.org/abs/2602.10364

[^64]: https://github.com/StanfordMIMI/Comp2Comp

[^65]: https://www.themoonlight.io/en/review/comp2comp-open-source-software-with-fda-cleared-artificial-intelligence-algorithms-for-computed-tomography-image-analysis

[^66]: https://arxiv.org/pdf/2602.10364.pdf

[^67]: https://comp2comp.readthedocs.io

[^68]: https://github.com/orgs/StanfordMIMI/repositories

[^69]: https://arxiv.org/html/2602.10364v1

[^70]: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices

[^71]: https://www.nature.com/articles/s41746-025-01800-1

[^72]: https://www.linkedin.com/posts/stephenaylward_opensource-monai-medicalimaging-activity-7183025822323113985-UesQ

[^73]: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_wholebody_ct_segmentation

[^74]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11482175/

[^75]: https://academic.oup.com/bioinformatics/article/41/2/btaf024/7958575

[^76]: https://dev.to/aleksanderobuchowski/medimageinsight-open-source-medical-image-embedding-model-now-on-huggingface-3b63

[^77]: https://www.jmir.org/2026/1/e81116

[^78]: https://x.com/BoWang87/status/1823041928948261119?lang=en

[^79]: https://github.com/HaneenElyamani/ECG-classification

[^80]: https://x.com/BoWang87/status/1823041928948261119?lang=en-GB

[^81]: https://github.com/helme/ecg_ptbxl_benchmarking

[^82]: https://huggingface.co/wanglab/ecg-fm/blob/9f1473ad22db10cfd15c6f4c13bcae48c353bd66/README.md

[^83]: https://arxiv.org/html/2507.09887v1

[^84]: https://huggingface.co/wanglab/ecg-fm/commit/584219ea492cdeef2e19ffbdf9c6ecc874ba427e

[^85]: https://huggingface.co/google/vit-base-patch16-224

[^86]: https://github.com/PKUDigitalHealth/ECGFounder

[^87]: https://huggingface.co/PKUDigitalHealth/ECGFounder

[^88]: https://huggingface.co/docs/transformers/en/model_doc/vit

[^89]: https://huggingface.co/docs/hub/en/repositories-licenses

[^90]: https://github.com/ml-jku/EVA

[^91]: https://github.com/Shaz-5/oral-cancer-classification

[^92]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9546660/

[^93]: https://ngdc.cncb.ac.cn/databasecommons/database/id/5435

[^94]: https://zenodo.org/records/14801921/files/HF_Model_Cards_December_2024_lic_and_arXiv.csv?download=1

[^95]: https://huggingface.co/codewithdark/vit-chest-xray/blob/main/README.md

[^96]: https://github.com/PrateekDutta2001/Oral-Cancer-Detection

[^97]: https://www.nature.com/articles/s41597-026-06736-z

[^98]: https://aikosh.indiaai.gov.in/home/datasets/details/oral_cancer_imaging_and_clinical_dataset.html

[^99]: https://huggingface.co/hustvl/vitmatte-small-distinctions-646

[^100]: https://arxiv.org/html/2405.05237v1

[^101]: https://github.com/maxium0526/cft-chexpert

[^102]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11861387/

[^103]: https://github.com/mahmoodlab/MIL-Lab

[^104]: https://github.com/DearCaat/RRT-MIL

[^105]: https://github.com/binli123/dsmil-wsi/blob/master/dsmil.py

[^106]: https://arxiv.org/html/2509.03903v1

[^107]: https://arxiv.org/html/2410.19877v3

[^108]: https://github.com/physionetchallenges/python-cnn-example-2024/blob/main/README.md

[^109]: https://blog.csdn.net/qq_56039091/article/details/151968125

[^110]: https://github.com/bowang-lab/ECG-FM/activity

[^111]: https://git.zib.de/bzfweima/ecg-jepa

[^112]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12858047/

[^113]: https://arxiv.org/html/2410.19877v2

[^114]: https://docs.nvidia.com/nemo/rl/0.2.1/design-docs/checkpointing.html

[^115]: https://huggingface.co/docs/huggingface_hub/guides/model-cards

[^116]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8018482/

[^117]: https://etasr.com/index.php/ETASR/article/view/17472

[^118]: https://verl.readthedocs.io/en/latest/advance/checkpoint.html

[^119]: https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/model-cards.md

[^120]: https://arxiv.org/abs/2310.02486

[^121]: https://huggingface.co/docs/accelerate/en/usage_guides/checkpoint

[^122]: https://arxiv.org/pdf/2506.21803.pdf

[^123]: https://github.com/hustvl/EVA-X/actions

[^124]: https://github.com/baaivision/EVA/issues/54

[^125]: https://ports.macports.org/port/clamav/builds/?page=4\&page=7

[^126]: https://huggingface.co/hustvl/vitmatte-base-composition-1k

[^127]: https://evadb.readthedocs.io/en/v0.2.6/source/reference/udfs/hf.html

[^128]: https://github.com/guanjinquan/OSCC-PathologyImageDataset

[^129]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11633071/

[^130]: http://arxiv.org/abs/2507.16360

[^131]: https://aikosha.indiaai.gov.in/home/datasets/details/oral_cancer_imaging_and_clinical_dataset.html

[^132]: https://www.kaggle.com/code/shivam17299/oral-cancer-lips-and-tongue-images-dataset

[^133]: https://arxiv.org/html/2507.16360v1

[^134]: https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images

[^135]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8516863/

[^136]: https://groups.google.com/g/physionet-challenges/c/6P3yZGS4zLM

[^137]: https://www.physionet.org/files/challenge-2021/1.0.2/papers/CinC2021-134.pdf

[^138]: https://www.sciencedirect.com/science/article/pii/S1568494622004446

[^139]: https://snucm.elsevierpure.com/en/publications/learning-ecg-representations-for-multi-label-classification-of-ca

[^140]: https://physionet.org/content/cpsc2021/

[^141]: https://moody-challenge.physionet.org/2021/results/

[^142]: https://moody-challenge.physionet.org/2021/

[^143]: https://github.com/awerdich/physionet

[^144]: https://physionet.org/content/challenge-2021/1.0.0/

[^145]: https://cinc.org/2021/Program/accepted/9_Preprint.pdf

[^146]: https://www.kaggle.com/datasets/bjoernjostein/physionet-challenge-2021-snomed-mappings

