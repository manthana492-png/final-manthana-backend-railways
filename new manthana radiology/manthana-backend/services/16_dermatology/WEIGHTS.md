# Dermatology service — model weights and hybrid pipeline

The dermatology service always uses **OpenRouter** (role `dermatology` in `config/cloud_inference.yaml`) for the **narrative** report. **Condition scores** (`pathology_scores` / `DERM_CLASSES`) come from a **priority chain** when weight files are present under **`MODEL_DIR`** (default `/models`).

## Modes ([`config.py`](./config.py))

| Variable | Purpose |
|----------|---------|
| `MODEL_DIR` | Directory searched for checkpoints (Modal Volume mount at `/models` recommended). |
| `DERM_HAM_CHECKPOINT` | Filename for HAM7 EfficientNet-V2-M weights (default `derm_efficientnet_v2m_ham7.pt`). |
| `derm_efficientnet_b4.pt` | Legacy B4 12-class head (fixed name via `CHECKPOINT_FILENAME` in config). |
| `DERM_CLASSIFIER_PRIORITY` | Comma-separated: `ham_v2`, `b4`, `openrouter` — first **available** branch wins (default `ham_v2,b4,openrouter`). |
| `DERM_HAM7_CLASS_ORDER` | Seven HAM class keys in **logit order** for the HAM checkpoint (must match training). Default: `akiec,bcc,bkl,df,mel,nv,vasc`. |
| `DERM_GRADCAM` | `1` / `true` — best-effort Grad-CAM PNG (base64) in `structures.derm_gradcam_png_base64` when a local PyTorch classifier ran; failures are ignored. |

## HAM7 → 12-class API

Raw seven probabilities are exposed as **`structures.ham10000_scores`**. Mapped probabilities still fill the legacy **`DERM_CLASSES`** keys in **`pathology_scores`** via [`ham_map.py`](./ham_map.py) (deterministic matrix). **Legal:** HAM10000 **data** is CC BY-NC; use only weight artifacts whose **license fits your product** (e.g. Apache 2.0 distributions). Document provenance in your compliance pack.

## Checkpoint format

1. **HAM (preferred when file exists and priority allows):** `torchvision.models.efficientnet_v2_m`, final linear **`out_features=7`**, `state_dict` load with `strict=False`. Preprocess: **224×224**, ImageNet normalize (see [`ham_classifier.py`](./ham_classifier.py)).
2. **B4:** `torchvision.models.efficientnet_b4`, custom head → **12** classes (`DERM_CLASSES`). Preprocess: **380×380** (see [`classifier.py`](./classifier.py)).

## Modal volume

From `manthana-backend` (adjust volume name if yours differs):

```bash
modal volume put manthana-model-weights ./path/to/derm_efficientnet_v2m_ham7.pt derm_efficientnet_v2m_ham7.pt
modal volume put manthana-model-weights ./path/to/derm_efficientnet_b4.pt derm_efficientnet_b4.pt
```

**Rollback:** remove or rename a checkpoint on the volume; the next priority tier (or OpenRouter scores) is used automatically.

## Response fields (observability)

- `structures.classifier_mode` — `efficientnet_v2m_ham7` | `efficientnet_b4` | `openrouter_vision_v1`
- `structures.ham10000_scores` — present when the HAM branch ran
- `structures.derm_gradcam_png_base64` — optional attention map (not histopathology)
- `models_used` — e.g. `EfficientNet-V2-M-HAM7` + narrative model slug
- `GET /ready` and `GET /health` include **`component_health`** (weight flags, resolved mode, priority list)

## Clinical

Screening / decision support only — not an FDA-cleared or CE-marked device unless obtained separately. Attention maps are explanatory overlays, not tissue diagnosis.
