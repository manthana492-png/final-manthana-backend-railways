# ECG service — no Modal / volume weights

The ECG service does **not** use downloadable neural-network checkpoints. **Do not** mount a Modal volume or upload `.pt` / founder files for ECG.

## What runs locally

| Component | Role |
|-----------|------|
| **Manthana-ECG-Engine** | Heuristic rhythm scores from the waveform (`shared/ecg_rhythm.py`) |
| **neurokit2** | PR / QRS / QT / QTc / HR (`shared/ecg_intervals.py`) |
| **Digitization** | OpenCV path or optional `ECG_DIGITISER_REPO_ROOT` adapter for ECG photos |
| **OpenRouter** | Role **`narrative_ecg`** in `config/cloud_inference.yaml` — prompt file `prompts/ecg_system.md` |

## Environment (no `MODEL_DIR` required)

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | Required for long-form narrative; core rhythm/interval output still works without it |
| `CLOUD_INFERENCE_CONFIG_PATH` | Optional; defaults to baked `cloud_inference.yaml` on Modal/Railway |
| `ECG_DIGITISER_REPO_ROOT` | Optional external PhysioNet-style digitiser |
| `ECG_IMAGE_QUALITY_GATE` | Photo quality warnings before digitization |

## Modal

`modal deploy modal_app/deploy_ecg.py` — **CPU only**, **no** `models_volume()` mount. Secrets: `OPENROUTER_API_KEY`, `MANTHANA_LLM_REPO_ROOT` as in other CPU services.

## Clinical

Screening / decision support only — not an FDA-cleared or CE-marked device unless obtained separately.
