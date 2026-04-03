# Open-Source AI/ML Medical Imaging Models with FDA Connection (March 2026)

## Executive overview

Systematic review of the FDA’s AI/ML-enabled medical device list, 510(k)/De Novo summaries, and major open-source model repositories finds **only one family of AI/ML imaging pipelines whose open-source code and weights are explicitly tied to cleared FDA devices and licensed under a clearly permissive license: the Comp2Comp AAQ and BMD pipelines used in Bunkerhill AAQ and Bunkerhill BMD.**[^1][^2][^3][^4][^5]
No other FDA-cleared AI/ML imaging device could be confirmed to publish both its production model weights and source code under a permissive (Apache/MIT/BSD) license as of March 2026.[^6][^1]

There is, however, a rich ecosystem of **non–FDA-cleared, open-source medical AI models**—notably TotalSegmentator, MONAI-based models, TorchXRayVision, and EVA-X—that either (a) are explicitly used as components in FDA-cleared products, or (b) have strong external validation in Radiology/Nature Medicine/npj Digital Medicine and could serve as building blocks in a commercial platform.[^7][^8][^9][^10][^11][^12][^13][^14][^15][^16]

The tables below focus on:

1. **FDA-cleared deep-learning pipelines with open-source implementations and permissive licenses.**  
2. **Open-source models and frameworks that are documented as components within FDA-cleared devices.**  
3. **Highly validated, open-source imaging models in major journals that are not yet FDA-cleared.**

Because regulatory filings rarely discuss licensing and most commercial vendors keep models proprietary, this list is best interpreted as **"all models that could be confidently verified from public evidence", not an absolutely complete global list.**

***

## 1. Radiology – CT opportunistic screening (FDA-cleared, open-source pipelines)

### 1.1 Comp2Comp AAQ (Abdominal Aortic Quantification) – Bunkerhill AAQ

| Attribute | Details |
| --- | --- |
| Medical specialty | Vascular radiology / abdominal CT |
| FDA device name | **Bunkerhill Abdominal Aortic Quantification (AAQ)** (software-only SaMD) |
| FDA pathway / number | 510(k) **K243779**; Class II, product code QIH (Medical Image Management and Processing System).[^4] |
| FDA clearance date | FDA decision letter and 510(k) summary dated **July 1, 2025**.[^4] |
| Cleared indication / intended use | Radiological image processing software to analyze CT exams with or without contrast that include the L1–L5 abdominal aorta region in adults ≥22 years; assists trained specialists by providing **maximum axial abdominal aortic diameter measurement** for normal and aneurysmal aortas; not for post‑operative aortas and not for stand‑alone clinical decision-making.[^4] |
| Open-source pipeline name | **Comp2Comp AAQ** within the StanfordMIMI **Comp2Comp** library.[^2][^17][^18] |
| Model architecture | Multi-stage 3D pipeline built around **nnU‑Net** for aorta segmentation and **TotalSegmentator nnU‑Net spine model** for L1–L5 localization/cropping; ellipse fitting per axial slice with selection of largest minor axis as maximum diameter.[^8][^19] |
| Repository URL | GitHub: `https://github.com/StanfordMIMI/Comp2Comp`.[^17][^20] |
| License | StanfordMIMI org metadata and Comp2Comp paper state **Apache License 2.0**, permitting commercial use, modification, and redistribution.[^21][^3]  Documentation site still lists GPLv3, so due‑diligence with maintainers is recommended.[^22] |
| Imaging modality | Abdominal **CT** (contrast and non-contrast), DICOM input.[^4][^2] |
| GPU / VRAM requirements | Pipeline relies on nnU‑Net and TotalSegmentator; nnU‑Net documentation and TotalSegmentator experience indicate **≈11–12 GB VRAM minimum**, with 16 GB GPUs recommended for full‑resolution 3D segmentation; can fall back to CPU with substantial slowdown.[^23][^24][^25] |
| CUDA / hardware | Implemented in **PyTorch/TensorFlow with CUDA**; Comp2Comp issue tracker notes TotalSegmentator “works only with CUDA-compatible GPUs” and uses `nvidia-smi` to query free memory.[^25] |
| Clinical validation (stand-alone) | Comp2Comp AAQ evaluated on 258 CT scans from four external institutions enriched for AAA (36% prevalence); **mean absolute error 1.57–1.58 mm** vs radiologist consensus, within predefined 2.0 mm margin; high agreement with radiologists (ICC ≈0.98, Pearson r ≈0.98) and Bland–Altman limits ±≈5 mm.[^2][^8][^19] 510(k) summary reports MAE 1.58 mm (95% CI 1.38–1.80).[^4] |
| Role in FDA device | Bunkerhill AAQ 510(k) summary describes a deep‑learning algorithm for maximal abdominal aortic diameter; Stanford and Bunkerhill communications explicitly state it is built on the **open-source Comp2Comp platform**, and moonlight review details the AAQ pipeline architecture matching the open-source code.[^8][^18][^26][^27][^28] |

### 1.2 Comp2Comp BMD (Bone Mineral Density) – Bunkerhill BMD

| Attribute | Details |
| --- | --- |
| Medical specialty | Musculoskeletal radiology / osteoporosis screening from CT |
| FDA device name | **Bunkerhill BMD** (Bone Mineral Density) |
| FDA pathway / number | 510(k) **K242295**; Regulation 21 CFR 892.1170 (Bone densitometer), Class II, product code KGI.[^5] |
| FDA clearance date | FDA decision letter dated **April 8, 2025**.[^5] |
| Cleared indication / intended use | Post-processing AI software for adults **≥30 years**, estimating DXA‑measured average areal bone mineral density of spinal bones from existing CT scans and outputting a **flag for low BMD below a threshold**; opportunistic assessment using prior or concurrent CT, without phantom; not a replacement for DXA.[^5][^29][^30][^31] |
| Open-source pipeline name | **Comp2Comp BMD** pipeline in the same library.[^2][^17] |
| Model architecture | Multi‑stage pipeline: nnU‑Net segmentation of L1–L4 vertebrae; ROI extraction for trabecular bone; calibration using visceral adipose tissue and air ROIs; regression and binary classifier on calibrated HU to predict low vs normal BMD; implemented in PyTorch.[^2][^32][^19] |
| Repository URL | GitHub: `https://github.com/StanfordMIMI/Comp2Comp` (same as AAQ).[^17] |
| License | Comp2Comp paper and Stanford MIMI org listing: **Apache 2.0**, explicitly described as permitting commercial and non‑commercial use; lawyers should verify against any residual GPLv3 references in older docs.[^21][^3][^22] |
| Imaging modality | Non‑contrast **abdominal CT** scans containing T12–L5 region; DICOM input.[^33][^5] |
| GPU / VRAM requirements | Same nnU‑Net / TotalSegmentator backbone and Comp2Comp implementation as AAQ; practical deployments generally require **≥11–12 GB VRAM** for 3D inference, with CPU‑only fallback.[^23][^24][^25] |
| Clinical validation | In Comp2Comp paper, evaluated on **371 CT scans** with concurrent DXA across four institutions; **sensitivity 81.0% (95% CI 74.0–86.8)** and **specificity 78.4% (95% CI 72.3–83.7)** for low BMD classification, AUROC ≈0.88, Pearson r ≈0.79 between continuous BMD estimate and DXA T‑score.[^2][^32][^19] 510(k) summary confirms same metrics and subgroup analyses with consistent performance across scanners, slice thickness and sites.[^5] |
| Role in FDA device | Bunkerhill press and Stanford Radiology articles state that the cleared BMD algorithm was developed from the **open-source Comp2Comp software** and validated within the Bunkerhill consortium.[^33][^29][^30][^34] |

### 1.3 Interpretation against your criteria

* **Open-source with code and weights** – Comp2Comp provides full source and model weights for AAQ and BMD; accompanying paper is explicit that these are *"fully open‑sourced, FDA‑510(k)-cleared deep learning pipelines"*.[^2][^35][^19]
* **Permissive license** – GitHub organization metadata and the paper specify **Apache 2.0**, which allows commercial SaaS deployment without copyleft obligations.  Some documentation pages still list GPLv3, so formal legal confirmation from maintainers is recommended before productization.[^22][^21][^3][^36][^37]
* **Commercial use allowed** – Apache 2.0 and similar permissive licenses explicitly allow for‑profit use, sublicensing, and integration into proprietary systems provided attribution and license text are preserved.[^36][^38]
* **GPU‑compatible** – Both pipelines are implemented in PyTorch/TensorFlow on CUDA, tested on NVIDIA GPUs; nnU‑Net and TotalSegmentator document GPU dependence and typical VRAM needs.[^23][^24][^25]
* **Medical imaging / diagnostics** – Both are radiology CT pipelines for clinical tasks (AAA sizing; opportunistic osteoporosis screening) and are authorized as FDA AI/ML-enabled SaMDs.[^4][^5][^1][^6]

Given current evidence, these are the **only pipelines that cleanly satisfy all of your hard constraints while being linked to specific FDA-cleared devices.**

***

## 2. Radiology – CT segmentation models used inside FDA-cleared products

These models are **not themselves FDA-cleared devices**, but there is explicit public documentation that they are used as components within cleared SaMD.

### 2.1 TotalSegmentator (CT; some MRI) – component in Bunkerhill AAQ and other devices

| Attribute | Details |
| --- | --- |
| Medical specialty | Cross-sectional radiology; whole‑body CT (and MRI in newer variants) |
| Model name / architecture | **TotalSegmentator** – 3D nnU‑Net ensemble segmenting 104+ anatomical structures in CT; separate **TotalSegmentator MRI** nnU‑Net for 80 structures.[^12][^39] |
| Repository URL | GitHub: `https://github.com/wasserth/TotalSegmentator`; PyPI: `TotalSegmentator` package.[^12][^40][^41] |
| License | **Apache License 2.0** for the software; authors note that v2 adds some new labels (e.g., appendicular bones, tissue types, heart chambers high‑res, face) whose *results* are licensed for non-commercial/non-clinical use only, while the original v1 label set and many v2 labels remain usable for commercial purposes.[^42][^10][^43][^44][^45] |
| Imaging modalities | CT (main model, 104–117 classes); separate MRI model (80 structures) released in Radiology.[^12][^39] |
| FDA relationship | Moonlight review of Comp2Comp confirms that the AAQ pipeline uses **TotalSegmentator’s spine nnU‑Net model** as the L1–L5 localization stage; hence TotalSegmentator is an internal component of the FDA‑cleared Bunkerhill AAQ device.[^8][^19]  Another 510(k) (K252054) cites training and validation using the TotalSegmentator dataset for sacrum segmentation, indicating use of TotalSegmentator data and methodology in additional cleared products.[^46] |
| GPU / VRAM | Authors and Slicer integration report that full‑resolution 3D inference typically needs **≈12 GB GPU memory**, with recommendation for ≥16 GB VRAM; some modes can run on CPU or in “fast” mode with lower resolution.[^23][^24][^47] |
| Clinical validation | Radiology AI paper and follow‑up studies show high Dice scores across 104 structures, and a 2024 Journal of Imaging Informatics in Medicine paper reports high intra‑individual reproducibility across 34 abdominal structures, often outperforming human readers and a separate nnU‑Net model.[^48][^11][^49][^50] |

**Implication:** TotalSegmentator itself is not FDA-cleared, but **its models and dataset are directly embedded in at least one cleared device (Bunkerhill AAQ) and used as training/validation data in others**, making it a strong candidate as an upstream component in a commercial platform, subject to compliance with the Apache 2.0 license and the specific restrictions on certain subclasses.

### 2.2 MONAI framework and MONAI Model Zoo

| Attribute | Details |
| --- | --- |
| Medical specialty | Generic – CT, MRI, ultrasound; multiple organs and pathologies |
| Component name | **MONAI** (Medical Open Network for AI) framework and **MONAI Model Zoo** bundles.[^7][^51] |
| Repositories / registries | GitHub: `Project-MONAI/monai`, `Project-MONAI/model-zoo`; NVIDIA NGC hosts multiple MONAI bundles such as whole-body CT segmentation built on TotalSegmentator data.[^7][^51][^52] |
| License | Core MONAI and Model Zoo are **Apache License 2.0**; bundles are encouraged but not required to follow Apache 2.0; many do.[^7][^51] |
| FDA relationship | 510(k) K232613 (**CT Cardiomegaly**) explicitly states that the subject device uses a non‑adaptive machine-learning algorithm implemented with the **MONAI framework**, contrasting it with its AI-based predicate that uses an “unknown framework”.[^9][^1][^53]  LinkedIn commentary from MONAI leadership notes other cleared devices citing MONAI as an “off-the-shelf” framework.[^9] |
| GPU / VRAM | Depends on specific bundle; NVIDIA NGC whole‑body CT models trained on TotalSegmentator data use SegResNet and expect modern NVIDIA GPUs with sufficient VRAM (varies with resolution; high‑res model intended for 16–24 GB+ VRAM).[^52] |
| Clinical validation | Individual MONAI-based research models are published widely, but the **framework itself** is a toolkit; safety and effectiveness are established at the device level (e.g., CT Cardiomegaly’s validation of cardiothoracic ratio measurements with high agreement to radiologists).[^1][^9] |

**Implication:** MONAI is a **permissive, GPU-native backbone** that is already named in FDA 510(k) summaries; building your own devices on MONAI aligns with precedents, but the specific models from the public Model Zoo are *not themselves* FDA-cleared or labeled for diagnostic use.[^51][^52][^7]

***

## 3. Radiology – Chest X‑ray (non–FDA‑cleared, open-source with strong validation)

These models are not yet associated with FDA clearances but satisfy your open‑source and permissive‑license constraints and have substantial external validation in chest imaging.

### 3.1 TorchXRayVision pre-trained models

| Attribute | Details |
| --- | --- |
| Medical specialty | Thoracic radiology – chest X‑ray classification and representation learning |
| Model family | **TorchXRayVision** pre‑trained CNNs (e.g., DenseNet‑121, ResNet variants) for multi‑label chest pathology prediction and feature extraction.[^13][^14] |
| Repository / registry | GitHub: `mlmed/torchxrayvision`; PyPI package `torchxrayvision`; multiple models also mirrored on Hugging Face.[^54][^14][^55][^56] |
| License | PyPI and third‑party registries list **Apache‑2.0** license.[^57][^54][^58] |
| Imaging modality | 2D chest X‑ray (PA/AP views) |
| GPU / VRAM | Standard 2D CNNs; practical deployments run comfortably on **8–12 GB VRAM** GPUs at batch sizes typical for inference; CPU inference is possible with lower throughput (not explicitly specified but follows from DenseNet scale). |
| Clinical validation | TorchXRayVision paper (MIDL 2022 and arXiv) reports performance across multiple public CXR datasets; models trained on combinations of NIH, CheXpert, MIMIC‑CXR etc. achieve AUROCs in the **0.8–0.9 range** for several pathologies; the library is widely used as a baseline and feature extractor in radiology research.[^13][^59][^60][^56]  Several Radiology AI and related papers use TorchXRayVision embeddings for downstream tasks like severity scoring and clinical trajectory prediction.[^59] |

### 3.2 EVA‑X chest X‑ray foundation model

| Attribute | Details |
| --- | --- |
| Medical specialty | Thoracic radiology – chest diseases from X‑ray |
| Model family | **EVA‑X** – family of Vision Transformers pre‑trained with self‑supervised masked image modeling plus tokenizer-based feature matching on >520k unlabeled chest X‑rays; architectures ViT‑Ti/16, ViT‑S/16, ViT‑B/16.[^61][^62][^16] |
| Repository | GitHub: `hustvl/EVA-X`; pre‑trained checkpoints linked from npj Digital Medicine article and arXiv preprint.[^61][^62][^15][^16] |
| License | Repository does not surface a standard SPDX license file in the root; public materials describe EVA‑X as “fully open-source, with pretrained models and a plug-and-play codebase”, but **the precise license (MIT/Apache/BSD vs custom) could not be confirmed from current documentation**, so legal review is required.[^63][^64] |
| Imaging modality | 2D chest X‑ray |
| GPU / VRAM | ViT‑Ti (~6M params) to ViT‑B (~86M); training uses multi‑GPU clusters; inference with ViT‑S/B typically fits in **8–16 GB VRAM** depending on batch size; the paper emphasizes data efficiency rather than minimal hardware.[^15][^65] |
| Clinical validation | npj Digital Medicine article reports EVA‑X achieving **state-of-the-art performance across 10–11 chest X‑ray tasks**, including disease classification, label‑efficient learning, and segmentation; for some tasks, small EVA‑X‑Ti surpasses larger supervised baselines and reaches ~95% COVID‑19 detection accuracy using only 1% labeled data.[^61][^15][^16] |

### 3.3 Other open CXR models and platforms

* **OpenCXR (Radboud DIAG)** – collection of open-source chest‑X‑ray algorithms and utilities (segmentation, classification); repository points to multiple models with open weights; licensing varies but many components are MIT or Apache.[^66]
* **MedicalPatchNet** – patch-based, self-explainable CXR classifier published in Scientific Reports (Nature portfolio) with code released on GitHub under an open-source license; matches EfficientNetV2 performance (AUROC ≈0.91) while improving localization interpretability.[^67]
* **MI2RLNet** – Korean radiology group’s platform providing open code and pre-trained weights for multiple radiology tasks; licenses differ by sub-project and require per‑model review; some are aimed at research only.[^68]

None of the above are currently documented as components in FDA-cleared products, but they satisfy your open‑source, GPU, and clinical‑validation requirements.

***

## 4. Radiology – Other CT/MR models (non–FDA‑cleared, high validation)

### 4.1 TotalSegmentator MRI

* **Task:** Segmentation of 80 anatomical structures across multiple MRI sequences; robust to sequence type.[^39]
* **Publication:** Radiology (2025) reports internal and external Dice scores ≈0.84 for 80 structures and superior performance to other open models; strong generalization across AMOS and CHAOS datasets.[^39]
* **License:** Same Apache‑2.0 base as TotalSegmentator CT, with similar commercial-use caveats for a few high‑res or sensitive label sets.[^42][^43][^44]
* **Relevance:** Not yet tied to FDA submissions but natural companion to CT‑based TotalSegmentator for multimodality segmentation.

### 4.2 NVIDIA NGC / MONAI whole-body CT segmentation models

* **Task:** SegResNet-based 3D segmentation of 104 whole-body structures, trained on the TotalSegmentator dataset; available as MONAI bundle on NVIDIA NGC.[^52]
* **License:** Bundle itself is distributed under an NVIDIA EULA with references to original Apache‑licensed TotalSegmentator dataset; commercial use terms depend on NGC license language and must be reviewed separately.[^52]
* **Validation:** Dice≈0.93 on TotalSegmentator dataset according to NGC documentation; not peer-reviewed to the same level as Radiology AI paper but leverages that dataset and architecture.[^52]

***

## 5. Cardiology – CT cardiothoracic ratio (FDA-cleared device using open framework)

While no fully open-source, FDA-cleared *cardiac imaging* models with public weights were identified, one cleared device explicitly uses an open-source framework.

### 5.1 CT Cardiomegaly (Innolitics) – MONAI-based device

| Attribute | Details |
| --- | --- |
| Medical specialty | Cardiology / thoracic radiology – cardiothoracic ratio from CT |
| FDA device name | **CT Cardiomegaly** |
| FDA pathway / number | 510(k) **K232613**, Class II, QIH.[^1][^9] |
| Indication | Command‑line SaMD that analyzes CT images containing the heart to compute linear and area‑based cardiothoracic ratio (CTR) using a **non‑adaptive machine learning algorithm**, assisting physicians but not replacing clinical judgment.[^1][^9] |
| Framework | 510(k) summary states that CT Cardiomegaly uses **non-adaptive ML algorithms implemented in the MONAI framework**, contrasting with its predicate’s AI algorithm that used an “unknown framework”.[^9] |
| Licensing | MONAI is Apache‑2.0; CT Cardiomegaly itself is proprietary. There is no evidence that its production weights are published. |

**Implication:** There is currently **no evidence** that CT Cardiomegaly’s trained weights are open-source, but it strengthens the regulatory precedent for **using Apache‑licensed MONAI as the backbone of FDA-cleared AI imaging devices.**

***

## 6. Other domains (pathology, ECG, ultrasound, ophthalmology, dermatology)

Extensive searching across the FDA AI/ML device list, PubMed, and major open-source hubs (GitHub, Hugging Face, PhysioNet, MONAI Zoo, NGC, Grand Challenge, TorchXRayVision, PyPI) **did not reveal any pathology, ECG, ultrasound, ophthalmology, or dermatology AI models that simultaneously meet all of your constraints and are clearly tied to an FDA 510(k)/De Novo authorization with public model weights.**[^69][^70][^1][^6]

Plenty of open-source models exist in these areas (e.g., digital pathology classifiers, retinal disease classifiers, dermatology skin-lesion CNNs), and many FDA-cleared commercial products exist, but **they are nearly all closed-source at the model level**; any use of open-source components is typically at the framework/library layer (e.g., TensorFlow, PyTorch, MONAI) rather than in the primary diagnostic model whose weights are disclosed.

***

## 7. Practical takeaways for your SaaS radiology platform

1. **Comp2Comp AAQ and BMD currently provide the only clear path to building on top of FDA-cleared, open-source AI pipelines for imaging.**  They match your requirements most closely and come with detailed validation mirroring FDA submissions.[^3][^19][^5][^2][^4]
2. **TotalSegmentator and MONAI are strategically important enabling technologies.**  They are permissively licensed, GPU-native, widely validated, and already appear as components or frameworks in FDA-cleared devices, but are not devices themselves.[^8][^9][^7][^51][^52]
3. **High-performing, open-source CXR models like TorchXRayVision and EVA‑X can power front-end triage and decision support** in your commercial platform today, but you will still need your own regulatory pathway if you claim diagnostic indications; they are not “pre-cleared” despite strong performance and publication track records.[^13][^14][^15][^16]
4. **Licensing diligence is essential.**  Even when a repository is Apache‑2.0, datasets may be CC‑BY‑NC or have non-commercial clauses; some TotalSegmentator v2 sub-models and MRI datasets, for example, restrict clinical or commercial use.  Comp2Comp’s GPL‑vs‑Apache discrepancy between docs and repo metadata likewise warrants direct confirmation.[^10][^43][^44][^45][^42]
5. **Most FDA-cleared AI imaging products remain closed-source.**  To stay aligned with your open-source, pay‑per‑use SaaS strategy, expect to combine **open-source, clinically validated building blocks** (Comp2Comp, TotalSegmentator, MONAI, TorchXRayVision, EVA‑X) with **your own regulatory submissions** rather than relying on turnkey FDA‑cleared, open‑weights models in most specialties.

---

## References

1. [Artificial Intelligence-Enabled Medical Devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices) - The AI-Enabled Medical Device List is a resource intended to identify AI-enabled medical devices tha...

2. [Comp2Comp: Open-Source Software with FDA-Cleared Artificial Intelligence Algorithms for Computed Tomography Image Analysis](https://www.arxiv.org/abs/2602.10364) - Artificial intelligence allows automatic extraction of imaging biomarkers from already-acquired radi...

3. [Comp2Comp: Open-Source Software with FDA-Cleared Artificial ...](https://arxiv.org/html/2602.10364v1) - Comp2Comp has a permissive Apache License 2.0 that allows for use, modification, and distribution fo...

4. [[PDF] BunkerHill Health John Smith, JD Partner Hogan Lovells US, L.L.P. ...](https://www.accessdata.fda.gov/cdrh_docs/pdf24/K243779.pdf) - Bunkerhill Abdominal Aortic Quantification (AAQ). Classification Name. Medical image management and ...

5. [[PDF] April 8, 2025 BunkerHill Health Eren Alkan Director of AI Algorithms ...](https://www.accessdata.fda.gov/cdrh_docs/pdf24/K242295.pdf) - bone mineral density of spinal bones from existing CT scans and outputs a flag for low bone density ...

6. [How AI is used in FDA-authorized medical devices](https://www.nature.com/articles/s41746-025-01800-1) - by R Singh · 2025 · Cited by 22 — The clinician-AI interface: intended use and explainability in FDA...

7. [Project-MONAI/model-zoo](https://github.com/Project-MONAI/model-zoo) - MONAI Model Zoo hosts a collection of medical imaging models in the MONAI Bundle format. All source ...

8. [[Revue de papier] Comp2Comp: Open-Source Software with FDA ...](https://www.themoonlight.io/fr/review/comp2comp-open-source-software-with-fda-cleared-artificial-intelligence-algorithms-for-computed-tomography-image-analysis) - This paper introduces Comp2Comp, an open-source software platform featuring two FDA 510 ... Abdomina...

9. [#opensource #monai #medicalimaging #ai | Stephen Aylward](https://www.linkedin.com/posts/stephenaylward_opensource-monai-medicalimaging-activity-7183025822323113985-UesQ) - The #opensource #MONAI platform for #medicalimaging #ai has enabled hundreds of applications and res...

10. [TotalSegmentator](https://totalsegmentator.com) - A running demo of the TotalSegmentator which can segment 117 anatomical structures in CT images

11. [TotalSegmentator: A Gift to the Biomedical Imaging Community | Radiology: Artificial Intelligence](https://pubs.rsna.org/doi/full/10.1148/ryai.230235)

12. [TotalSegmentator: robust segmentation of 104 anatomical ...](https://arxiv.org/abs/2208.05868v1) - In this work we focus on automatic segmentation of multiple anatomical structures in (whole body) CT...

13. [TorchXRayVision: A library of chest X-ray datasets and models](https://proceedings.mlr.press/v172/cohen22a/cohen22a.pdf) - by JP Cohen · 2022 · Cited by 225 — Abstract. TorchXRayVision is an open source software library for...

14. [TorchXRayVision: A library of chest X-ray datasets and models - arXiv](https://arxiv.org/abs/2111.00595) - TorchXRayVision is an open source software library for working with chest X-ray datasets and deep le...

15. [EVA-X: a foundation model for general chest x-ray analysis ...](https://www.nature.com/articles/s41746-025-02032-z) - by J Yao · 2025 · Cited by 19 — EVA-X is a family of medical foundational models pre-trained specifi...

16. [EVA-X: a foundation model for general chest x-ray analysis with self ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12624024/) - Here, we present EVA-X, an innovative foundational model based on X-ray images with broad applicabil...

17. [Computed tomography to body composition (Comp2Comp).](https://github.com/StanfordMIMI/Comp2Comp) - Comp2Comp is a library for extracting clinical insights from computed tomography scans. Installation...

18. [Newest FDA-Cleared Innovation Turns Routine CTs Into an Early ...](https://med.stanford.edu/radiology/news/2025-news/newest-fda-cleared-innovation-turns-routine-cts-into-an-early-wa.html) - Using the open-source Comp2Comp image-analysis platform developed by Dr. Chaudhari's lab, the AAQ al...

19. [[Literature Review] Comp2Comp: Open-Source Software with FDA ...](https://www.themoonlight.io/en/review/comp2comp-open-source-software-with-fda-cleared-artificial-intelligence-algorithms-for-computed-tomography-image-analysis) - This paper introduces Comp2Comp, an open-source software platform featuring two FDA 510(k)-cleared d...

20. [Comp2Comp/README.md at main · StanfordMIMI/Comp2Comp](https://github.com/StanfordMIMI/Comp2Comp/blob/main/README.md) - Computed tomography to body composition (Comp2Comp). - StanfordMIMI/Comp2Comp

21. [Stanford MIMI Lab](https://github.com/orgs/StanfordMIMI/repositories) - Stanford MIMI Lab has 20 repositories available. Follow their code on GitHub.

22. [Welcome to comp2comp's documentation! — comp2comp ...](https://comp2comp.readthedocs.io) - Welcome to comp2comp's documentation! . Comp2Comp . License: GPL v3 GitHub Workflow Status Documenta...

23. [TotalSegmentator results computed in fast mode are rough - Support](https://discourse.slicer.org/t/totalsegmentator-results-computed-in-fast-mode-are-rough/28768) - ... TotalSegmentator: Tool for robust segmentation of 104 important anatomical structures in CT imag...

24. [Add option to force using CPU · Issue #37 - GitHub](https://github.com/wasserth/TotalSegmentator/issues/37) - Problem: People who have a weak NVIDIA GPU ... Then TotalSegmentator will not find cuda inside of py...

25. [Local Implementation @ AppleSilicon M1 · Issue #30 - GitHub](https://github.com/StanfordMIMI/Comp2Comp/issues/30) - Comp2Comp requires tensorflow, TotalSegmentator PyTorch. The ... This will not work on a M1/M2 machi...

26. [New FDAClearance for Abdominal Aortic Quantification Algorithm](https://www.linkedin.com/posts/stanfordradiology_newest-fda-cleared-innovation-turns-routine-activity-7354594919493066755-83jI) - 𝗡𝗲𝘄 𝗙𝗗𝗔-𝗖𝗹𝗲𝗮𝗿𝗲𝗱 𝗜𝗻𝗻𝗼𝘃𝗮𝘁𝗶𝗼𝗻 𝗧𝘂𝗿𝗻𝘀 𝗥𝗼𝘂𝘁𝗶𝗻𝗲 𝗖𝗧𝘀 𝗜𝗻𝘁𝗼 𝗮𝗻 𝗘𝗮𝗿𝗹𝘆-𝗪𝗮𝗿𝗻𝗶𝗻𝗴 𝗦𝘆𝘀𝘁𝗲𝗺 𝗳𝗼𝗿 𝗔𝗻𝗲𝘂𝗿𝘆𝘀𝗺𝘀 A cross-disc...

27. [Stanford team's A4 AAA algorithm cleared by FDA as Bunkerhill AAQ](https://www.linkedin.com/posts/bunkerhill-health_big-milestone-for-the-stanford-university-led-activity-7348719752581681154-_r18) - ... 510(k) ... The development of the Stanford A4 AAA Algorithm (also known as AAQ) began at Stanfor...

28. [Second 510(k) clearance for Comp2Comp platform - LinkedIn](https://www.linkedin.com/posts/akshaychaudhari_hot-of-the-heels-of-the-first-fda-clearance-activity-7346896836885942273-e37a) - We're excited to announce that Bunkerhill Health has received FDA clearance for Bunkerhill Abdominal...

29. [Bunkerhill Health Achieves FDA Clearance for ...](https://www.bunkerhillhealth.com/resources/bunkerhill-bmd-clearance) - Part of Bunkerhill’s Carebricks platform, Bunkerhill BMD helps care teams identify and navigate foll...

30. [Bunkerhill Health Achieves FDA Clearance for Opportunistic Bone Density Assessment on Routine Abdominal CT](https://kalkinemedia.com/news/world-news/bunkerhill-health-achieves-fda-clearance-for-opportunistic-bone-density-assessment-on-routine-abdominal-ct) - Bunkerhill Health Achieves FDA Clearance for Opportunistic Bone Density Assessment on Routine Abdomi...

31. [Opportunistic screening tool for osteoporosis nabs FDA clearance](https://healthimaging.com/topics/medical-imaging/diagnostic-screening/opportunistic-screening-tool-osteoporosis-nabs-fda-clearance) - The algorithm can be applied to noncontrast abdominal CT scans completed for any clinical indication...

32. [Comp2Comp: Open-Source Software with FDA-Cleared ...](https://arxiv.org/pdf/2602.10364.pdf) - The Comp2Comp pipeline includes the first FDA 510(k)-cleared, open-source deep learning so- lutions ...

33. [Clinical Translation of an AI Tool to Detect Osteoporosis on ...](https://med.stanford.edu/radiology/news/2025-news/clinical-translation-of-an-ai-tool-to-detect-osteoporosis-on-rou.html) - The FDA-cleared algorithm is intended for use in adults 30 years and older, to use existing abdomina...

34. [Congrats to Bunkerhill Health on another FDA cleared algorithm ...](https://www.linkedin.com/posts/nikipezeshki_congrats-to-bunkerhill-health-on-another-activity-7325279702888173569-kLq5) - Congrats to Bunkerhill Health on another FDA cleared algorithm! Nishith Khandwala David Eng

35. [Comp2Comp: Open-Source Software with FDA-Cleared Artificial ...](http://paperreading.club/page?id=376472) - Artificial intelligence allows automatic extraction of imaging biomarkers from already-acquired radi...

36. [Apache 2.0 License - Using Creative Commons and Open Software ...pitt.libguides.com › openlicensing › apache2](https://pitt.libguides.com/openlicensing/apache2) - Balancing the rights of creators and users, open licenses grant users some permissions to use and di...

37. [LICENSE-2.0.txt - Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt)

38. [Apache License - Wikipedia](https://en.wikipedia.org/wiki/Apache_License)

39. [TotalSegmentator MRI: Robust Sequence-independent Segmentation of Multiple Anatomic Structures in MRI | Radiology](https://pubs.rsna.org/doi/10.1148/radiol.241613) - TotalSegmentator MRI is an open-source, user-friendly tool that provides automatic, robust sequence-...

40. [wasserth/TotalSegmentator: Tool for robust segmentation ...](https://github.com/wasserth/TotalSegmentator) - Tool for robust segmentation of >100 important anatomical structures in CT and MR images - wasserth/...

41. [TotalSegmentator - PyPI](https://pypi.org/project/TotalSegmentator/) - TotalSegmentator 2.13.0. pip install TotalSegmentator. Copy PIP instructions ... License: Apache 2.0...

42. [TotalSegmentator/LICENSE at master](https://github.com/wasserth/TotalSegmentator/blob/master/LICENSE) - A permissive license whose main conditions require preservation of copyright and license notices. Co...

43. [TotalSegmentator v2 - Announcements](https://discourse.slicer.org/t/totalsegmentator-v2/32470) - People should be aware that while the previous version of TotalSegmentator was freely available, the...

44. [YongchengYAO/TotalSegmentator-MR-Lite · Datasets at Hugging ...](https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-MR-Lite) - About. This is a derivative of the TotalSegmentator dataset. 616 MR images and corresponding segment...

45. [YongchengYAO/TotalSegmentator-CT-Lite · Datasets at Hugging Face](https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite) - This is a derivative of the TotalSegmentator dataset. 1228 CT images and corresponding segmentation ...

46. [[PDF] K252054 - Kevin Murrock - accessdata.fda.gov](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K252054.pdf) - Data Source: The model was trained using relevant scans from the TotalSegmentator dataset which incl...

47. [GitHub - lassoan/SlicerTotalSegmentator: Fully automatic total body ...](https://github.com/lassoan/SlicerTotalSegmentator) - 3D Slicer extension for fully automatic whole body CT segmentation using "TotalSegmentator" AI model...

48. [TotalSegmentator: A Gift to the Biomedical Imaging Communitypmc.ncbi.nlm.nih.gov › articles › PMC10546367](https://pmc.ncbi.nlm.nih.gov/articles/PMC10546367/)

49. [Intra-Individual Reproducibility of Automated Abdominal ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092311/) - The purpose of this study is to assess segmentation reproducibility of artificial intelligence-based...

50. [Intra‐Individual Reproducibility of Automated Abdominal ...](https://d-nb.info/1349935921/34)

51. [MONAI Model Zoo](https://rocm.docs.amd.com/projects/monai/en/latest/reference/model-zoo.html) - The MONAI Model Zoo is a hub for researchers and data scientists to share, discover, and deploy the ...

52. [GPU-optimized AI, Machine Learning, & HPC Software | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_wholebody_ct_segmentation) - This model is trained using the SegResNet [3] network. The model is trained using TotalSegmentator d...

53. [ViG/LICENSE at main · hustvl/ViG](https://github.com/hustvl/ViG/blob/main/LICENSE) - MIT License. A short and simple permissive license with conditions only requiring preservation of co...

54. [torchxrayvision](https://pypi.org/project/torchxrayvision/) - License. OSI Approved :: Apache Software License ; Operating System. OS Independent ; Programming La...

55. [torchxrayvision/densenet121-res224-pc](https://huggingface.co/torchxrayvision/densenet121-res224-pc) - Here is how to use this model to classify an image of xray: Note: Each pretrained model has 18 outpu...

56. [Introduction — TorchXRayVision 1.0.1 documentation](https://mlmed.org/torchxrayvision/)

57. [pypi.org : torchxrayvision](https://packages.ecosyste.ms/registries/pypi.org/packages/torchxrayvision) - ... torchxrayvision , transfer-learning. License: Apache-2.0. Latest release: 2 months ago. First re...

58. [balavenkatesh3322/CV-pretrained-model: A collection of computer ...](https://github.com/balavenkatesh3322/CV-pretrained-model) - Apache License 2.0 · FastPhotoStyle, A Closed-form Solution to Photorealistic Image Stylization. PyT...

59. [[PDF] TorchXRayVision: A library of chest X-ray datasets and models - arXiv](https://arxiv.org/pdf/2111.00595.pdf)

60. [TorchXRayVision: A library of chest X-ray datasets and models](https://proceedings.mlr.press/v172/cohen22a.html) - TorchXRayVision is an open source software library for working with chest X-ray datasets and deep le...

61. [EVA-X: A Foundation Model for General Chest X-ray ...](https://arxiv.org/abs/2405.05237) - by J Yao · 2024 · Cited by 19 — Here we present EVA-X, an innovative foundational model based on X-r...

62. [EVA-X: A Foundation Model for General Chest X-ray ...](https://ui.adsabs.harvard.edu/abs/2024arXiv240505237Y/abstract) - The diagnosis and treatment of chest diseases play a crucial role in maintaining human health. X-ray...

63. [Bo Wang's Post - LinkedIn](https://www.linkedin.com/posts/bo-wang-a6065240_tiny-models-massive-capacity-zero-labels-activity-7397320942059835392-ejCT) - Tiny Models. Massive Capacity. Zero Labels. The future of health AI is here!! I’m excited to share t...

64. [A data management system for precision medicine](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0000464) - by JJL Jacobs · 2025 · Cited by 5 — This paper evaluates a MedDMS in five types of use cases for pre...

65. [A foundation model for general chest X-ray analysis with ...](https://arxiv.org/html/2405.05237v1) - EVA-X is a family of medical foundational models pre-trained specifically for analyzing and diagnosi...

66. [DIAGNijmegen/opencxr: A collection of open-source ...](https://github.com/DIAGNijmegen/opencxr) - OpenCXR is an open source collection of chest x-ray (CXR) algorithms and utilities maintained by the...

67. [MedicalPatchNet: a patch-based self-explainable AI architecture for chest X-ray classification](https://www.nature.com/articles/s41598-026-40358-0) - Deep neural networks excel in radiological image classification but frequently suffer from poor inte

68. [An Open Medical Platform to Share Source Code and Various Pre-Trained Weights for Models to Use in Deep Learning Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC8628158/) - Deep learning-based applications have great potential to enhance the quality of medical services. Th...

69. [FDA's AI Medical Device List: Stats, Trends & Regulation](https://intuitionlabs.ai/articles/fda-ai-medical-device-tracker) - Learn about the FDA's AI/ML medical device tracker. With 1451 devices authorized through 2025 and 29...

70. [The state of artificial intelligence-based FDA-approved medical ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC7486909/) - At the beginning of the artificial intelligence (AI)/machine learning (ML) era, the expectations are...

