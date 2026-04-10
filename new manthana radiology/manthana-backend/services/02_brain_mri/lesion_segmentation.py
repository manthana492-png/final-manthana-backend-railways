"""
Pluggable brain lesion segmentation (BraTS-style).
Supports nnU-Net / ONNX / TorchScript models via env var BRAIN_LESION_MODEL_PATH.
Graceful fallback when model not deployed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.brain_mri.lesion")

_lesion_model: Any = None


def _load_lesion_model() -> Any | None:
    """Load brain lesion segmentation model if available."""
    global _lesion_model
    path = (os.environ.get("BRAIN_LESION_MODEL_PATH") or "").strip()
    if not path or not os.path.isfile(path):
        return None
    if _lesion_model is not None:
        return _lesion_model
    
    try:
        # Try TorchScript first
        import torch
        _lesion_model = torch.jit.load(path, map_location="cpu")
        _lesion_model.eval()
        logger.info("Loaded brain lesion segmentation TorchScript from %s", path)
        return _lesion_model
    except Exception as e:
        logger.debug("TorchScript load failed: %s", e)
    
    try:
        # Try ONNX
        import onnxruntime as ort
        _lesion_model = ort.InferenceSession(path)
        logger.info("Loaded brain lesion segmentation ONNX from %s", path)
        return _lesion_model
    except Exception as e:
        logger.debug("ONNX load failed: %s", e)
    
    logger.warning("Could not load lesion model from %s", path)
    return None


def _preprocess_for_lesion(
    volume: np.ndarray,
    target_shape: tuple[int, int, int] = (128, 128, 128),
) -> tuple[np.ndarray, tuple]:
    """Preprocess volume for lesion segmentation."""
    v = np.asarray(volume, dtype=np.float32)
    
    # Normalize intensity
    p1, p99 = np.percentile(v, [1, 99])
    v = np.clip(v, p1, p99)
    v = (v - v.mean()) / max(v.std(), 1e-6)
    
    original_shape = v.shape
    
    # Simple resize to target shape
    if v.shape != target_shape:
        import scipy.ndimage
        zoom_factors = [t / o for t, o in zip(target_shape, v.shape)]
        v = scipy.ndimage.zoom(v, zoom_factors, order=1)
    
    # Add batch and channel dimensions
    v = v[np.newaxis, np.newaxis, ...]
    
    return v, original_shape


def run_lesion_segmentation(
    volume: np.ndarray,
    contrasts: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Run brain lesion segmentation.
    
    Args:
        volume: Primary MRI volume (T1, T2, or FLAIR)
        contrasts: Dict with available contrasts (T1, T1ce, T2, FLAIR)
        
    Returns:
        {
            tumor_volume_ml,
            enhancing_volume_ml,
            edema_volume_ml,
            lesion_location,
            available,
            reason (if unavailable)
        }
    """
    model = _load_lesion_model()
    if model is None:
        return {
            "available": False,
            "reason": "model_not_configured",
            "tumor_volume_ml": None,
            "enhancing_volume_ml": None,
            "edema_volume_ml": None,
            "lesion_location": None,
        }
    
    try:
        # Preprocess
        input_vol, original_shape = _preprocess_for_lesion(volume)
        
        # Run inference
        if hasattr(model, "forward"):
            # TorchScript
            import torch
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_vol)
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                output = output.numpy()
        else:
            # ONNX
            input_name = model.get_inputs()[0].name
            output = model.run(None, {input_name: input_vol})[0]
        
        # Post-process: BraTS-style 4-class output
        # Class 0: background, 1: necrotic/core, 2: edema, 3: enhancing
        if output.shape[1] == 4:  # Multi-class
            pred = np.argmax(output[0], axis=0)
            
            # Resize back to original shape
            if pred.shape != original_shape:
                import scipy.ndimage
                zoom_factors = [o / t for o, t in zip(original_shape, pred.shape)]
                pred = scipy.ndimage.zoom(pred, zoom_factors, order=0)
            
            # Calculate volumes
            # Assume 1mm isotropic for simplicity; real implementation should use DICOM spacing
            voxel_volume_ml = np.prod(original_shape) / np.prod(original_shape)  # 1ml per voxel assumption
            
            necrotic_vol = float(np.sum(pred == 1)) * voxel_volume_ml
            edema_vol = float(np.sum(pred == 2)) * voxel_volume_ml
            enhancing_vol = float(np.sum(pred == 3)) * voxel_volume_ml
            tumor_vol = necrotic_vol + enhancing_vol
            
            # Determine lesion location (simplified)
            if np.any(pred > 0):
                centroid = np.mean(np.argwhere(pred > 0), axis=0)
                z, y, x = centroid
                z_mid, y_mid, x_mid = [s / 2 for s in pred.shape]
                
                location_parts = []
                if z < z_mid * 0.7:
                    location_parts.append("anterior")
                elif z > z_mid * 1.3:
                    location_parts.append("posterior")
                if y < y_mid:
                    location_parts.append("right")
                else:
                    location_parts.append("left")
                if x < x_mid * 0.7:
                    location_parts.append("inferior")
                elif x > x_mid * 1.3:
                    location_parts.append("superior")
                
                location = "_".join(location_parts) if location_parts else "central"
            else:
                location = None
            
            return {
                "available": True,
                "tumor_volume_ml": round(tumor_vol, 2),
                "enhancing_volume_ml": round(enhancing_vol, 2),
                "edema_volume_ml": round(edema_vol, 2),
                "lesion_location": location,
                "whole_tumor_volume_ml": round(tumor_vol + edema_vol, 2),
            }
        else:
            # Binary or single-output model
            pred = (output[0, 0] > 0.5).astype(np.uint8)
            
            if pred.shape != original_shape:
                import scipy.ndimage
                zoom_factors = [o / t for o, t in zip(original_shape, pred.shape)]
                pred = scipy.ndimage.zoom(pred, zoom_factors, order=0)
            
            voxel_volume_ml = 1.0  # Simplified
            tumor_vol = float(np.sum(pred)) * voxel_volume_ml
            
            return {
                "available": True,
                "tumor_volume_ml": round(tumor_vol, 2),
                "enhancing_volume_ml": None,
                "edema_volume_ml": None,
                "lesion_location": None,
                "whole_tumor_volume_ml": round(tumor_vol, 2),
            }
        
    except Exception as e:
        logger.warning("Lesion segmentation failed: %s", e)
        return {
            "available": False,
            "reason": "inference_failed",
            "tumor_volume_ml": None,
            "enhancing_volume_ml": None,
            "edema_volume_ml": None,
            "lesion_location": None,
        }