"""
Shared Modal helpers for Manthana GPU services (CT/MRI, X-ray, USG, pathology, cytology, mammography, lab_report,
**oral_cancer — production default**) and CPU scale-to-zero services (ECG, dermatology, optional **oral_cancer_cpu**).

Deploy from `manthana-backend` root:
  modal deploy modal_app/deploy_ct_brain.py

Layout expected on your machine:
  this_studio/
    config/cloud_inference.yaml
    packages/manthana-inference/
    new manthana radiology/manthana-backend/   <-- cwd for modal deploy
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

MODELS_VOLUME_NAME = os.environ.get("MANTHANA_MODAL_VOLUME", "manthana-model-weights")
MODAL_SECRET_NAME = os.environ.get("MANTHANA_MODAL_SECRET", "manthana-env")


def backend_root() -> Path:
    """manthana-backend directory."""
    return Path(__file__).resolve().parent.parent


def studio_root() -> Path:
    """this_studio (parent of new manthana radiology)."""
    return Path(__file__).resolve().parents[3]


def models_volume() -> modal.Volume:
    return modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)


def manthana_secret() -> modal.Secret:
    return modal.Secret.from_name(MODAL_SECRET_NAME)


def gpu_function_kwargs(
    *,
    timeout: int = 600,
    scaledown_window: int = 90,
    max_containers: int = 4,
) -> dict[str, Any]:
    return {
        "timeout": timeout,
        "scaledown_window": scaledown_window,
        "max_containers": max_containers,
    }


def cuda_runtime_python311() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])
        .apt_install(
            "git",
            "libgl1",
            "libglib2.0-0",
            "python3-dev",
        )
        .pip_install("pip", "setuptools", "wheel")
    )


def with_manthana_llm_stack(img: modal.Image) -> modal.Image:
    """packages/manthana-inference + cloud_inference.yaml + instructor (for llm_router)."""
    sr = studio_root()
    pkg = sr / "packages" / "manthana-inference"
    cfg = sr / "config" / "cloud_inference.yaml"
    if not pkg.is_dir():
        raise RuntimeError(
            f"manthana-inference not found at {pkg}. "
            "Clone this_studio with packages/ or set paths before modal deploy."
        )
    if not cfg.is_file():
        raise RuntimeError(f"cloud_inference.yaml not found at {cfg}")
    return (
        img.copy_local_dir(str(pkg), "/app/packages/manthana-inference")
        .copy_local_file(str(cfg), "/app/config/cloud_inference.yaml")
        .run_commands(
            "pip install -e /app/packages/manthana-inference",
            "pip install 'instructor>=1.0.0'",
        )
    )


def copy_shared(img: modal.Image, br: Path | None = None) -> modal.Image:
    br = br or backend_root()
    return img.copy_local_dir(str(br / "shared"), "/app/shared")


def with_monai_transforms(img: modal.Image) -> modal.Image:
    """MONAI medical imaging transforms (library only; no bundle download at image build)."""
    return img.pip_install("monai==1.3.0", "nibabel>=4.0")


def install_nnunet_v2(img: modal.Image) -> modal.Image:
    return img.pip_install(
        "git+https://github.com/MIC-DKFZ/nnUNet.git@v2.4.2#egg=nnunetv2",
    )


def install_comp2comp_opt(img: modal.Image) -> modal.Image:
    return img.run_commands(
        "git clone --depth 1 https://github.com/StanfordMIMI/Comp2Comp /opt/Comp2Comp",
        "cd /opt/Comp2Comp && pip install --no-cache-dir -e .",
    ).env({"COMP2COMP_DIR": "/opt/Comp2Comp"})


def service_image_ct_brain() -> modal.Image:
    br = backend_root()
    req = br / "services" / "11_ct_brain" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_monai_transforms(img)
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    return img.copy_local_dir(str(br / "services" / "11_ct_brain"), "/app").env(
        {"MANTHANA_USE_MONAI_CT_LOADER": "1"}
    )


def service_image_brain_mri() -> modal.Image:
    br = backend_root()
    req = br / "services" / "02_brain_mri" / "requirements.txt"
    img = cuda_runtime_python311()
    img = install_nnunet_v2(img)
    img = img.pip_install_from_requirements(str(req))
    img = img.pip_install("tensorflow>=2.15,<2.18")
    img = with_monai_transforms(img)
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    return img.copy_local_dir(str(br / "services" / "02_brain_mri"), "/app").env(
        {"MANTHANA_USE_MONAI_CT_LOADER": "1"}
    )


def service_image_cardiac_ct() -> modal.Image:
    br = backend_root()
    req = br / "services" / "04_cardiac_ct" / "requirements.txt"
    img = cuda_runtime_python311()
    img = install_nnunet_v2(img)
    img = img.pip_install_from_requirements(str(req))
    img = with_monai_transforms(img)
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = install_comp2comp_opt(img)
    return img.copy_local_dir(str(br / "services" / "04_cardiac_ct"), "/app").env(
        {"MANTHANA_USE_MONAI_CT_LOADER": "1"}
    )


def service_image_spine_neuro() -> modal.Image:
    br = backend_root()
    req = br / "services" / "10_spine_neuro" / "requirements.txt"
    img = cuda_runtime_python311()
    img = install_nnunet_v2(img)
    img = img.pip_install_from_requirements(str(req))
    img = with_monai_transforms(img)
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = install_comp2comp_opt(img)
    return img.copy_local_dir(str(br / "services" / "10_spine_neuro"), "/app").env(
        {"MANTHANA_USE_MONAI_CT_LOADER": "1"}
    )


def service_image_abdominal_ct() -> modal.Image:
    br = backend_root()
    req = br / "services" / "08_abdominal_ct" / "requirements.txt"
    img = cuda_runtime_python311()
    img = install_nnunet_v2(img)
    img = img.pip_install_from_requirements(str(req))
    img = with_monai_transforms(img)
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = install_comp2comp_opt(img)
    return img.copy_local_dir(str(br / "services" / "08_abdominal_ct"), "/app").env(
        {"MANTHANA_USE_MONAI_CT_LOADER": "1"}
    )


def service_image_ct_brain_vista() -> modal.Image:
    """VISTA-3D premium stack: same CT brain service + MONAI + optional foundation weights on volume."""
    br = backend_root()
    req = br / "services" / "11_ct_brain" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_monai_transforms(img)
    img = img.pip_install("huggingface_hub>=0.20.0")
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "11_ct_brain"), "/app")
    return img.env(
        {
            "MODEL_DIR": "/models",
            "VISTA3D_MODEL_PATH": "/models/vista3d/model.pt",
            "VISTA3D_ENABLED": "true",
            "MANTHANA_LLM_REPO_ROOT": "/app",
            "MANTHANA_USE_MONAI_CT_LOADER": "1",
        }
    )


def service_image_body_xray() -> modal.Image:
    """TorchXRayVision + OpenCV body_xray; TXRV weights baked in image; no nnUNet/Comp2Comp."""
    br = backend_root()
    req = br / "services" / "01_body_xray" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "01_body_xray"), "/app")
    img = img.run_commands(
        "python -c \""
        "import torchxrayvision as xrv; "
        "xrv.models.DenseNet(weights='densenet121-res224-all'); "
        "xrv.models.DenseNet(weights='densenet121-res224-chex'); "
        "xrv.models.DenseNet(weights='densenet121-res224-mimic_nb'); "
        "print('TXRV weights pre-download OK')"
        "\"",
    )
    # Railway gateway cannot serve heatmaps written inside this container
    return img.env({"XRAY_HEATMAP_URL_MODE": "none"})


def service_image_ultrasound() -> modal.Image:
    """Rad-DINO (microsoft/rad-dino) + OpenRouter narrative_usg; HF weights baked into image."""
    br = backend_root()
    req = br / "services" / "09_ultrasound" / "requirements.txt"
    cache = "/root/.manthana/models"
    img = cuda_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.env({"MANTHANA_MODEL_CACHE": cache})
    img = img.copy_local_dir(str(br / "services" / "09_ultrasound"), "/app")
    img = img.run_commands(
        f"mkdir -p {cache} && python -c \""
        "from transformers import AutoImageProcessor, AutoModel; "
        f"AutoImageProcessor.from_pretrained('microsoft/rad-dino', cache_dir='{cache}'); "
        f"AutoModel.from_pretrained('microsoft/rad-dino', cache_dir='{cache}'); "
        "print('Rad-DINO weights cached OK')\"",
    )
    return img


def service_image_pathology() -> modal.Image:
    """OpenSlide + Virchow/UNI2/ConvNeXt tile embeddings + DSMIL + vision_pathology narrative."""
    br = backend_root()
    req = br / "services" / "05_pathology" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.apt_install("libopenslide-dev", "libopenslide0")
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "05_pathology"), "/app")
    img = img.run_commands(
        "python -c \""
        "import timm; "
        "timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=0); "
        "print('timm ConvNeXt fallback weights cached OK')\"",
    )
    return img.env({"MODEL_DIR": "/models", "MANTHANA_MODEL_CACHE": "/models"})


def service_image_cytology() -> modal.Image:
    """Same embedding stack as pathology + DSMIL + cytology narrative role."""
    br = backend_root()
    req = br / "services" / "11_cytology" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.apt_install("libopenslide-dev", "libopenslide0")
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "11_cytology"), "/app")
    img = img.run_commands(
        "python -c \""
        "import timm; "
        "timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=0); "
        "print('timm ConvNeXt fallback weights cached OK')\"",
    )
    return img.env({"MODEL_DIR": "/models", "MANTHANA_MODEL_CACHE": "/models"})


def service_image_mammography() -> modal.Image:
    """Mirai (Lab-Rasool/Mirai or MIRAI_HF_REPO) + mammography narrative; use A10G if Mirai OOM on T4."""
    br = backend_root()
    req = br / "services" / "12_mammography" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "12_mammography"), "/app")
    return img.env({"MODEL_DIR": "/models", "DEVICE": "cuda"})


def service_image_lab_report() -> modal.Image:
    """MedGemma (shared/medical_document_parser) + OpenRouter lab_report role; use A10G if OOM on T4."""
    br = backend_root()
    req = br / "services" / "15_lab_report" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "15_lab_report"), "/app")
    return img.env(
        {
            "MODEL_DIR": "/models",
            "MANTHANA_MODEL_CACHE": "/models",
            "DEVICE": "cuda",
        }
    )


def service_image_oral_cancer() -> modal.Image:
    """EfficientNet-B3 + optional UNI + OpenRouter oral_cancer; B3 processor warmed at build."""
    br = backend_root()
    req = br / "services" / "14_oral_cancer" / "requirements.txt"
    img = cuda_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "14_oral_cancer"), "/app")
    img = img.run_commands(
        "python -c \""
        "from transformers import EfficientNetImageProcessor; "
        "EfficientNetImageProcessor.from_pretrained('google/efficientnet-b3'); "
        "print('EfficientNet-B3 processor cache OK')\""
    )
    return img.env(
        {
            "MODEL_DIR": "/models",
            "MANTHANA_MODEL_CACHE": "/models",
            "DEVICE": "cuda",
        }
    )


# ────────────────────────────────────────────────────────────────────────────────
# CPU-only helpers for bursty low-cost inference (ECG, Dermatology, optional Oral CPU)
# Production oral screening with UNI → use deploy_oral_cancer.py (GPU). These scale to $0 idle.
# ────────────────────────────────────────────────────────────────────────────────


def cpu_function_kwargs(
    *,
    timeout: int = 300,
    scaledown_window: int = 60,
    max_containers: int = 3,
) -> dict[str, Any]:
    """Autoscaling settings for CPU-only containers (scale to zero, pay per use)."""
    return {
        "timeout": timeout,
        "scaledown_window": scaledown_window,
        "max_containers": max_containers,
    }


def cpu_runtime_python311() -> modal.Image:
    """CPU-only base image (no CUDA) — smaller, faster cold start than GPU images."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            "git",
            "libgl1",
            "libglib2.0-0",
            "libgomp1",
            "python3-dev",
        )
        .pip_install("pip", "setuptools", "wheel")
    )


def service_image_ecg_cpu() -> modal.Image:
    """ECG service — CPU-only; neurokit2 + heuristics + OpenRouter (no torch, no Modal volume)."""
    br = backend_root()
    req = br / "services" / "13_ecg" / "requirements.txt"
    img = cpu_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    return img.copy_local_dir(str(br / "services" / "13_ecg"), "/app").env(
        {
            "DEVICE": "cpu",
            "MANTHANA_LLM_REPO_ROOT": "/app",
        }
    )


def service_image_dermatology_cpu() -> modal.Image:
    """Dermatology service - CPU-only (OpenRouter vision + optional local V2 weights)."""
    br = backend_root()
    req = br / "services" / "16_dermatology" / "requirements.txt"
    img = cpu_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    return img.copy_local_dir(str(br / "services" / "16_dermatology"), "/app").env(
        {
            "DEVICE": "cpu",
            "MANTHANA_LLM_REPO_ROOT": "/app",
        }
    )


def service_image_oral_cancer_cpu() -> modal.Image:
    """Oral cancer — CPU-only alternate (low volume / cost); production default is GPU ``deploy_oral_cancer``."""
    br = backend_root()
    req = br / "services" / "14_oral_cancer" / "requirements.txt"
    img = cpu_runtime_python311()
    img = img.pip_install_from_requirements(str(req))
    img = with_manthana_llm_stack(img)
    img = copy_shared(img, br)
    img = img.copy_local_dir(str(br / "services" / "14_oral_cancer"), "/app")
    img = img.run_commands(
        "python -c \""
        "from transformers import EfficientNetImageProcessor; "
        "EfficientNetImageProcessor.from_pretrained('google/efficientnet-b3'); "
        "print('EfficientNet-B3 processor cache OK')\""
    )
    return img.env(
        {
            "MODEL_DIR": "/models",
            "MANTHANA_MODEL_CACHE": "/models",
            "DEVICE": "cpu",
            "MANTHANA_LLM_REPO_ROOT": "/app",
        }
    )
