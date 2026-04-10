# Oral cancer service — model weights

**Production (Modal):** deploy **`modal_app/deploy_oral_cancer.py`** (GPU **T4**, shared volume **`/models`**, **`memory=8192`** MiB). Set Railway **`ORAL_CANCER_SERVICE_URL`** to that HTTPS origin + **`/analyze/oral_cancer`**. Use **`deploy_oral_cancer_cpu.py`** only as a cost-saving alternate if you do not need **UNI** latency on GPU.

Screening scores should come from task-specific vision weights under `MODEL_DIR` (default `/models`). Narrative and structured JSON fallback use OpenRouter (`oral_cancer` role in `config/cloud_inference.yaml`).

## Files

| File | Role | Notes |
|------|------|--------|
| `oral_effnet_v2m.pt` | Primary clinical photo (torchvision EfficientNet-V2-M) | Prefer Apache-2.0–compatible checkpoints. Class order for 3-class heads: **0=Normal, 1=OPMD, 2=OSCC** (same as `CLASSES` in `inference.py`). |
| `oral_cancer_finetuned.pt` | Legacy clinical photo (Hugging Face EfficientNet-B3) | Used when present and ordering selects B3 before or after V2-M per `ORAL_PREFER_V2M`. |
| `uni_oral_linear_head.pt` | Optional linear head on UNI embeddings | Histopathology / H&E-style inputs. |

## Binary (2-class) V2-M checkpoints

If weights are **normal vs malignant** only, set:

- `ORAL_V2M_NUM_CLASSES=2`
- Optional: `ORAL_V2M_BINARY_OPMD_FRACTION` (default `0.45`) — share of the malignant probability mass mapped to **OPMD**; `1 - fraction` goes to **OSCC-suspicious**. This is a conservative display mapping, not a separate pathology classifier.

Index order must be **0 = non-malignant / normal**, **1 = malignant**.

## Environment

- `ORAL_EFFNET_V2M_CHECKPOINT` — filename under `MODEL_DIR` (default `oral_effnet_v2m.pt`).
- `ORAL_PREFER_V2M` — empty = V2-M tried before B3 when both exist; `0`/`false` = B3 first; `1`/`true` = V2-M first when V2-M weights exist.
- `MODEL_DIR` — mount or volume root containing the `.pt` files.

## Modal volume

Create/update the shared volume (name configurable via `MANTHANA_MODAL_VOLUME`, default `manthana-model-weights`):

```bash
cd manthana-backend
modal volume put manthana-model-weights ./path/to/oral_effnet_v2m.pt oral_effnet_v2m.pt
```

Redeploy `modal_app/deploy_oral_cancer.py` (GPU) or `deploy_oral_cancer_cpu.py` after uploading. Ensure secrets include `MODEL_DIR=/models`, `OPENROUTER_API_KEY`, and `ORAL_CANCER_ENABLED=true` as needed.

## Railway / Docker

Mount or bake weights into the path set by `MODEL_DIR` for the oral service (`services/14_oral_cancer/Dockerfile.railway`). Set gateway `ORAL_CANCER_SERVICE_URL` to `https://<host>/analyze/oral_cancer`.

## Rollback

Remove or rename a bad `.pt` on the volume; the pipeline falls back to the other EfficientNet branch if weights exist, then OpenRouter vision JSON if configured.
