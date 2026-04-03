"""
Tile embeddings for pathology/cytology: Virchow → UNI2 → ConvNeXt fallback.
Never raises — returns empty list on total failure.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.tile_embedding")


def ensure_hf_login() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as e:
        logger.debug("HF login skipped: %s", e)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _to_feature_vec(out: Any) -> np.ndarray:
    import torch

    if isinstance(out, torch.Tensor):
        t = out
    else:
        t = out.last_hidden_state.mean(dim=1) if hasattr(out, "last_hidden_state") else out
    if hasattr(t, "cpu"):
        t = t.cpu()
    a = np.asarray(t.detach().numpy() if hasattr(t, "detach") else t, dtype=np.float32).reshape(-1)
    return a


def _embed_transformers_virchow(lazy_model: Any, tiles: list, max_tiles: int) -> list[np.ndarray]:
    import torch

    ensure_hf_login()
    model = lazy_model.get()
    model.eval()
    dev = _device()
    embeddings: list[np.ndarray] = []
    for tile_img, _loc in tiles[:max_tiles]:
        tensor = (
            torch.tensor(tile_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        if dev == "cuda":
            tensor = tensor.cuda()
        with torch.no_grad():
            emb = model(tensor)
            if hasattr(emb, "last_hidden_state"):
                emb = emb.last_hidden_state.mean(dim=1)
            embeddings.append(np.asarray(emb.cpu().numpy(), dtype=np.float32).reshape(-1))
    return embeddings


def _imagenet_transform():
    import torchvision.transforms as T

    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _uni_transform():
    return _imagenet_transform()


def _virchow_transform():
    return _imagenet_transform()


def _forward_timm_features(model: Any, x: Any) -> Any:
    import torch

    if hasattr(model, "forward_features"):
        feat = model.forward_features(x)
    else:
        feat = model(x)
    if not isinstance(feat, torch.Tensor):
        return feat
    if feat.dim() == 4:
        feat = feat.mean(dim=(2, 3))
    elif feat.dim() == 3:
        feat = feat.mean(dim=1)
    return feat


def _embed_timm_tiles(model: Any, transform: Any, tiles: list, max_tiles: int) -> list[np.ndarray]:
    import torch

    dev = _device()
    model.eval()
    model.to(dev)
    embeddings: list[np.ndarray] = []
    for tile_img, _loc in tiles[:max_tiles]:
        x = transform(tile_img.astype(np.uint8)).unsqueeze(0).to(dev)
        with torch.no_grad():
            feat = _forward_timm_features(model, x)
            vec = _to_feature_vec(feat)
            embeddings.append(vec)
    return embeddings


def run_tile_embeddings(
    tiles: list,
    lazy_model: Any | None,
    max_tiles: int = 100,
) -> tuple[list[np.ndarray], str, str | None]:
    """
    Try transformers Virchow (lazy_model), timm Virchow, UNI2, ConvNeXt.
    Returns (embeddings, model_name, limitation_note or None).
    """
    ensure_hf_login()
    if not tiles:
        return [], "no-model", "No tiles to embed."

    # 1) HuggingFace transformers Virchow (existing LazyModel)
    if lazy_model is not None:
        try:
            embs = _embed_transformers_virchow(lazy_model, tiles, max_tiles)
            if embs:
                return embs, "Virchow", None
        except Exception as e:
            logger.warning("Virchow (transformers) unavailable: %s", e)

    try:
        import timm
        import torch
    except ImportError:
        logger.warning("timm/torch not available for embedding fallbacks")
        return [], "no-model", "timm/torch not installed."

    # 2) timm HuggingFace Virchow
    try:
        kwargs: dict[str, Any] = {"pretrained": True}
        try:
            from timm.layers import SwiGLUPacked

            kwargs["mlp_layer"] = SwiGLUPacked
            kwargs["act_layer"] = torch.nn.SiLU
        except Exception:
            pass
        model = timm.create_model("hf-hub:paige-ai/Virchow", **kwargs)
        embs = _embed_timm_tiles(model, _virchow_transform(), tiles, max_tiles)
        if embs:
            return embs, "Virchow", None
    except Exception as e:
        logger.warning("Virchow (timm) unavailable: %s", e)

    # 3) UNI2-h
    try:
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        embs = _embed_timm_tiles(model, _uni_transform(), tiles, max_tiles)
        if embs:
            return embs, "UNI2", "Virchow unavailable; using UNI2-h fallback."
    except Exception as e:
        logger.warning("UNI2 unavailable: %s", e)

    # 4) ConvNeXt (ImageNet)
    try:
        model = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
        )
        embs = _embed_timm_tiles(model, _imagenet_transform(), tiles, max_tiles)
        if embs:
            return (
                embs,
                "ConvNeXt-fallback",
                "Non-domain-specific ImageNet embeddings (ConvNeXt); interpret with caution.",
            )
    except Exception as e:
        logger.warning("ConvNeXt unavailable: %s", e)

    return [], "no-model", "No embedding model produced features."
