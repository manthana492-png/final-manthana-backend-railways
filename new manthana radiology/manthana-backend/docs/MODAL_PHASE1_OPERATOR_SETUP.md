# Phase 1 — Modal operator setup (one-time)

Run from `manthana-backend` root on a machine with the full `this_studio` tree (`packages/manthana-inference`, `config/cloud_inference.yaml`).

1. **Install CLI**
   ```bash
   pip install -r modal_requirements.txt
   modal token new
   ```

2. **Create Modal secret `manthana-env`** in the Modal dashboard with at least:
   - `OPENROUTER_API_KEY`
   - `MODEL_DIR=/models`
   - `MANTHANA_LLM_REPO_ROOT=/app`
   - Service weight paths (e.g. `CT_BRAIN_TORCHSCRIPT_PATH=/models/ct_brain/ich_main.pt`)

3. **Bootstrap TotalSegmentator weights on the volume**
   ```bash
   modal run modal_app/bootstrap_weights.py
   ```

4. **Optional — MONAI bundles + VISTA-3D weights**
   ```bash
   set MANTHANA_BOOTSTRAP=monai
   modal run modal_app/bootstrap_weights.py
   set MANTHANA_BOOTSTRAP=vista
   modal run modal_app/bootstrap_weights.py
   ```
   (On Unix: `MANTHANA_BOOTSTRAP=monai modal run modal_app/bootstrap_weights.py`)

5. **Upload proprietary weights**
   ```bash
   modal volume put manthana-model-weights ./weights/ich_main.pt ct_brain/ich_main.pt
   ```

See [modal_app/MODAL_DEPLOYMENT_FINAL.md](../modal_app/MODAL_DEPLOYMENT_FINAL.md) for the full deploy matrix.
