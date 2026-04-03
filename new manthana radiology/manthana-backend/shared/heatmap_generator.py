"""
Manthana — Heatmap Generator (Shared Utility)
Dual-strategy heatmap generation:
  1. Real Grad-CAM from PyTorch model activations
  2. Synthetic anatomical heatmap from findings + pathology_scores

Both produce a standard PNG overlay image.
"""

import os
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("manthana.heatmap")

HEATMAP_DIR = os.getenv("HEATMAP_DIR", "/tmp/manthana_uploads/heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)


# ═══ ANATOMICAL REGION MAPPING ═══
# Maps human-readable region strings → approximate (cy, cx, radius) as fractions of image
CHEST_REGIONS = {
    "right upper lobe":     (0.25, 0.35, 0.12),
    "right middle lobe":    (0.40, 0.35, 0.10),
    "right lower lobe":     (0.55, 0.35, 0.13),
    "left upper lobe":      (0.25, 0.65, 0.12),
    "left lower lobe":      (0.55, 0.65, 0.13),
    "right lung":           (0.40, 0.35, 0.18),
    "left lung":            (0.40, 0.65, 0.18),
    "right lung field":     (0.40, 0.35, 0.18),
    "left lung field":      (0.40, 0.65, 0.18),
    "cardiac silhouette":   (0.50, 0.52, 0.14),
    "heart":                (0.50, 0.52, 0.14),
    "mediastinum":          (0.35, 0.50, 0.10),
    "hilum":                (0.38, 0.50, 0.08),
    "right hilum":          (0.38, 0.42, 0.07),
    "left hilum":           (0.38, 0.58, 0.07),
    "diaphragm":            (0.70, 0.50, 0.16),
    "costophrenic angle":   (0.68, 0.35, 0.08),
    "aortic arch":          (0.25, 0.52, 0.08),
    "trachea":              (0.18, 0.50, 0.05),
    "central venous line":  (0.32, 0.48, 0.06),
    "pleural":              (0.45, 0.30, 0.15),
}

# Generic fallback for non-chest modalities
GENERIC_REGIONS = {
    "central":    (0.50, 0.50, 0.20),
    "upper":      (0.25, 0.50, 0.18),
    "lower":      (0.75, 0.50, 0.18),
    "left":       (0.50, 0.30, 0.18),
    "right":      (0.50, 0.70, 0.18),
}

SEVERITY_INTENSITY = {
    "critical": 1.0,
    "warning":  0.7,
    "info":     0.4,
    "clear":    0.15,
}


def _apply_jet_colormap(heatmap_gray: np.ndarray) -> np.ndarray:
    """Apply a JET-like colormap to a grayscale heatmap (0-255) → RGB."""
    h, w = heatmap_gray.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Normalized 0-1
    norm = heatmap_gray.astype(np.float32) / 255.0

    # JET colormap approximation
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0, 1)

    rgb[:, :, 0] = (r * 255).astype(np.uint8)
    rgb[:, :, 1] = (g * 255).astype(np.uint8)
    rgb[:, :, 2] = (b * 255).astype(np.uint8)

    return rgb


def _find_region_coords(region_str: str, detected_region: str = "chest"):
    """Map a region string to (cy, cx, radius) in fractional coords."""
    if not region_str:
        return (0.45, 0.50, 0.18)

    region_lower = region_str.lower().strip()

    # Try chest-specific regions first
    if detected_region in ("chest", "thorax"):
        for key, coords in CHEST_REGIONS.items():
            if key in region_lower:
                return coords

    # Try generic regions
    for key, coords in GENERIC_REGIONS.items():
        if key in region_lower:
            return coords

    # Default: center of image
    return (0.45, 0.50, 0.15)


def generate_synthetic_heatmap(
    image_path: str,
    findings: list,
    pathology_scores: dict,
    job_id: str,
    detected_region: str = "chest",
) -> str:
    """
    Generate a synthetic heatmap from findings and pathology scores.
    Uses Gaussian blobs placed at anatomical regions.
    
    Returns: relative URL path to the heatmap image.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed, cannot generate heatmap")
        return ""

    try:
        img = Image.open(image_path)
        w, h = img.size
    except Exception as e:
        logger.warning(f"Cannot open image for heatmap: {e}")
        w, h = 512, 512

    # Create empty heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Create coordinate grids
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy.astype(np.float32) / h
    xx = xx.astype(np.float32) / w

    # Add Gaussian blobs for each finding
    if isinstance(findings, list):
        for finding in findings:
            if isinstance(finding, dict):
                region = finding.get("region", "")
                severity = finding.get("severity", "info")
                confidence = finding.get("confidence", 50) / 100.0
            else:
                # Finding might be a string
                continue

            cy, cx, radius = _find_region_coords(region, detected_region)
            intensity = SEVERITY_INTENSITY.get(severity, 0.5) * confidence

            # Gaussian blob
            dist_sq = ((yy - cy) ** 2 + (xx - cx) ** 2) / (radius ** 2)
            blob = np.exp(-dist_sq * 2.0) * intensity
            heatmap = np.maximum(heatmap, blob)

    # Also add from pathology_scores
    for pathology, score in pathology_scores.items():
        if score < 0.3:
            continue
        region_name = pathology.lower()
        cy, cx, radius = _find_region_coords(region_name, detected_region)
        dist_sq = ((yy - cy) ** 2 + (xx - cx) ** 2) / (radius ** 2)
        blob = np.exp(-dist_sq * 2.0) * score * 0.5
        heatmap = np.maximum(heatmap, blob)

    # Normalize to 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = np.zeros((h, w), dtype=np.uint8)

    # Apply JET colormap
    heatmap_rgb = _apply_jet_colormap(heatmap)

    # Create RGBA with alpha from intensity
    alpha = (heatmap.astype(np.float32) / 255.0 * 200).astype(np.uint8)  # max 200/255 alpha
    heatmap_rgba = np.dstack([heatmap_rgb, alpha])

    # Save as PNG
    out_filename = f"{job_id}_heatmap.png"
    out_path = os.path.join(HEATMAP_DIR, out_filename)

    try:
        heatmap_img = Image.fromarray(heatmap_rgba, "RGBA")
        heatmap_img.save(out_path, "PNG")
        logger.info(f"[{job_id}] Synthetic heatmap saved: {out_path}")
        return f"/heatmaps/{out_filename}"
    except Exception as e:
        logger.error(f"[{job_id}] Failed to save heatmap: {e}")
        return ""


def generate_gradcam_heatmap(
    model,
    target_layer,
    input_tensor,
    image_path: str,
    job_id: str,
    target_category: int = None,
) -> str:
    """
    Generate a real Grad-CAM heatmap from a PyTorch model.
    
    Requires: pytorch-grad-cam (pip install grad-cam)
    
    Args:
        model: PyTorch model (eval mode)
        target_layer: The convolutional layer to extract CAM from
        input_tensor: Preprocessed input tensor [1, C, H, W]
        image_path: Original image path (for sizing)
        job_id: Unique job identifier
        target_category: Which class to generate CAM for (None = top prediction)
    
    Returns: relative URL path to the heatmap image.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from PIL import Image
    except ImportError:
        logger.warning("pytorch-grad-cam not installed, falling back to synthetic")
        return ""

    try:
        # Set up GradCAM
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Generate CAM
        targets = None  # Will use top prediction
        if target_category is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            targets = [ClassifierOutputTarget(target_category)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # First image in batch

        # Load original image for sizing
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        img_np = np.array(img.resize((grayscale_cam.shape[1], grayscale_cam.shape[0])))
        img_float = img_np.astype(np.float32) / 255.0

        # Create overlay
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # Resize back to original
        cam_pil = Image.fromarray(cam_image).resize((w, h), Image.LANCZOS)

        # Also create alpha-channel version for overlay
        cam_resized = np.array(
            Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
        )
        cam_rgb = _apply_jet_colormap(cam_resized)
        alpha = (cam_resized.astype(np.float32) / 255.0 * 200).astype(np.uint8)
        cam_rgba = np.dstack([cam_rgb, alpha])

        # Save
        out_filename = f"{job_id}_heatmap.png"
        out_path = os.path.join(HEATMAP_DIR, out_filename)

        heatmap_img = Image.fromarray(cam_rgba, "RGBA")
        heatmap_img.save(out_path, "PNG")
        logger.info(f"[{job_id}] Grad-CAM heatmap saved: {out_path}")
        return f"/heatmaps/{out_filename}"

    except Exception as e:
        logger.error(f"[{job_id}] Grad-CAM generation failed: {e}")
        return ""


def generate_heatmap(
    image_path: str,
    job_id: str,
    findings: list,
    pathology_scores: dict,
    detected_region: str = "chest",
    model=None,
    target_layer=None,
    input_tensor=None,
) -> str:
    """
    Unified heatmap generator — tries real Grad-CAM first, falls back to synthetic.
    
    Returns: relative URL path (e.g., "/heatmaps/{job_id}_heatmap.png")
    """
    # Strategy 1: Try real Grad-CAM if model is provided
    if model is not None and target_layer is not None and input_tensor is not None:
        result = generate_gradcam_heatmap(
            model=model,
            target_layer=target_layer,
            input_tensor=input_tensor,
            image_path=image_path,
            job_id=job_id,
        )
        if result:
            return result
        logger.info(f"[{job_id}] Grad-CAM failed, falling back to synthetic heatmap")

    # Strategy 2: Synthetic heatmap from findings
    return generate_synthetic_heatmap(
        image_path=image_path,
        findings=findings,
        pathology_scores=pathology_scores,
        job_id=job_id,
        detected_region=detected_region,
    )
