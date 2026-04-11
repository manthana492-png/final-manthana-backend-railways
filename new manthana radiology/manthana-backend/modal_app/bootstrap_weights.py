"""
One-off weight bootstrap into the Modal Volume.

Run from manthana-backend (with Modal CLI authenticated):
  modal run modal_app/bootstrap_weights.py

Optional (env ``MANTHANA_BOOTSTRAP``):
  MANTHANA_BOOTSTRAP=monai  — MONAI Model Zoo bundles under /models/monai_bundles
  MANTHANA_BOOTSTRAP=vista — VISTA-3D / foundation checkpoints under /models/vista3d

Default (unset or ``totalseg``): TotalSegmentator open tasks only (**``total``**, **``total_mr``**).
Tasks like **``vertebrae_body``** / **``heartchambers``** need a TotalSegmentator academic or commercial
license (`https://backend.totalsegmentator.com/license-academic/`). Set **``TOTALSEG_LICENSE_NUMBER``**
(18-character key from that site) in the environment before ``modal run``, and optionally
**``MANTHANA_TOTALSEG_BOOTSTRAP_TASKS=total,total_mr,vertebrae_body``** to extend the list.
"""

from __future__ import annotations

import os

import modal

# Do not import modal_app.common: ``modal run`` loads this file alone in the worker
# (/root/bootstrap_weights.py) without the modal_app package on PYTHONPATH.
# Keep MODELS_VOLUME_NAME / models_volume() aligned with modal_app/common.py.
MODELS_VOLUME_NAME = os.environ.get("MANTHANA_MODAL_VOLUME", "manthana-model-weights")
MODAL_SECRET_NAME = os.environ.get("MANTHANA_MODAL_SECRET", "manthana-env")


def models_volume() -> modal.Volume:
    return modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)


app = modal.App("manthana-bootstrap-weights")

_bootstrap_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pip", "setuptools", "wheel")
    .pip_install("TotalSegmentator>=2.4.0", "nibabel", "numpy")
)

_monai_torch_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pip", "setuptools", "wheel")
    .pip_install(
        "monai==1.3.0",
        "nibabel",
        "numpy",
        "torch>=2.1,<2.5",
        "huggingface_hub>=0.20.0",
    )
)


@app.function(
    image=_bootstrap_image,
    volumes={"/models": models_volume()},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=3600,
    cpu=8.0,
    memory=16384,
)
def bootstrap_totalsegmentator_weights() -> dict:
    import numpy as np
    import nibabel as nib

    os.makedirs("/models/_bootstrap", exist_ok=True)
    nii_path = "/models/_bootstrap/tiny.nii.gz"
    data = np.zeros((48, 48, 32), dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), nii_path)
    os.environ["MODEL_DIR"] = "/models"

    from totalsegmentator.config import set_license_number, setup_totalseg
    from totalsegmentator.python_api import totalsegmentator

    setup_totalseg()
    lic = (os.environ.get("TOTALSEG_LICENSE_NUMBER") or "").strip()
    if lic:
        # Offline length check inside TotalSegmentator; avoids network during image build.
        set_license_number(lic, skip_validation=True)

    raw = (os.environ.get("MANTHANA_TOTALSEG_BOOTSTRAP_TASKS") or "total,total_mr").strip()
    tasks = [t.strip() for t in raw.split(",") if t.strip()]

    done: list[str] = []
    errors: dict[str, str] = {}
    for task in tasks:
        out_dir = f"/models/_bootstrap/out_{task.replace(' ', '_')}"
        os.makedirs(out_dir, exist_ok=True)
        use_fast = task in ("total", "total_mr")
        try:
            totalsegmentator(
                nii_path,
                out_dir,
                task=task,
                fast=use_fast,
                device="cpu",
            )
            done.append(task)
        except BaseException as e:  # noqa: BLE001 — TotalSegmentator calls sys.exit on license errors
            code = getattr(e, "code", None)
            suffix = f" (code={code!r})" if code is not None else ""
            errors[task] = (f"{type(e).__name__}: {e}{suffix}")[:500]

    return {
        "volume": MODELS_VOLUME_NAME,
        "tasks_ok": done,
        "tasks_failed": errors,
        "hint": (
            "Upload CT brain / WMH / lesion weights with: modal volume put ... "
            "| Licensed TotSeg tasks: set TOTALSEG_LICENSE_NUMBER then "
            "MANTHANA_TOTALSEG_BOOTSTRAP_TASKS=total,total_mr,vertebrae_body,heartchambers"
        ),
    }


@app.function(
    image=_monai_torch_image,
    volumes={"/models": models_volume()},
    timeout=3600,
    cpu=4.0,
    memory=8192,
)
def bootstrap_monai_bundles() -> dict:
    from monai.bundle import download

    os.makedirs("/models/monai_bundles", exist_ok=True)
    bundles = ["swin_unetr_btcv"]
    done: list[str] = []
    errors: dict[str, str] = {}
    for b in bundles:
        try:
            download(name=b, bundle_dir="/models/monai_bundles", source="github")
            done.append(b)
        except Exception as e:  # noqa: BLE001
            errors[b] = str(e)[:500]
    return {"volume": MODELS_VOLUME_NAME, "bundles_ok": done, "bundles_failed": errors}


@app.function(
    image=_monai_torch_image,
    volumes={"/models": models_volume()},
    timeout=3600,
    cpu=4.0,
    memory=8192,
)
def bootstrap_vista3d_weights() -> dict:
    """Download VISTA-style checkpoint if available on Hugging Face (repo layout may change)."""
    os.makedirs("/models/vista3d", exist_ok=True)
    from huggingface_hub import hf_hub_download, snapshot_download

    candidates = [
        ("nvidia/vista3d", "model.pt"),
        ("nvidia/VISTA3D", "model.pt"),
    ]
    errors: list[str] = []
    for repo_id, filename in candidates:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="/models/vista3d",
            )
            return {
                "volume": MODELS_VOLUME_NAME,
                "vista3d_path": path,
                "repo_id": repo_id,
                "hint": "Set VISTA3D_MODEL_PATH in manthana-env if path differs.",
            }
        except Exception as e:  # noqa: BLE001
            errors.append(f"{repo_id}/{filename}: {str(e)[:400]}")

    try:
        snap = snapshot_download(
            repo_id="nvidia/vista3d",
            local_dir="/models/vista3d",
            ignore_patterns=["*.md", "*.txt"],
        )
        return {
            "volume": MODELS_VOLUME_NAME,
            "vista3d_snapshot_dir": snap,
            "note": "Inspect directory for .pt / .pth checkpoints; set VISTA3D_MODEL_PATH accordingly.",
            "hf_errors": errors,
        }
    except Exception as e2:  # noqa: BLE001
        return {
            "volume": MODELS_VOLUME_NAME,
            "error": str(e2)[:500],
            "hf_errors": errors,
            "hint": "Upload weights manually: modal volume put ... /models/vista3d/",
        }


@app.local_entrypoint()
def main():
    mode = os.environ.get("MANTHANA_BOOTSTRAP", "totalseg").strip().lower()
    if mode == "monai":
        print(bootstrap_monai_bundles.remote())
    elif mode in ("vista", "vista3d", "vista-full", "premium-ct"):
        if mode in ("vista-full", "premium-ct"):
            print("Downloading full VISTA-3D 127-class foundation model...")
        print(bootstrap_vista3d_weights.remote())
        if mode in ("vista-full", "premium-ct"):
            print("Premium CT ready for 127-class segmentation")
    else:
        print(bootstrap_totalsegmentator_weights.remote())
