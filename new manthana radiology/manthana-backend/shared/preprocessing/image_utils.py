"""
Manthana — Image Preprocessing Utilities
CLAHE, resize, normalize, grayscale, auto-enhance for medical images.
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger("manthana.image_utils")


def load_image(filepath: str) -> np.ndarray:
    """Load an image from file as RGB numpy array."""
    from PIL import Image
    img = Image.open(filepath)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Essential for camera photos of X-rays taken from lightboxes —
    enhances contrast and normalizes brightness.
    """
    import cv2

    if len(image.shape) == 3:
        image = to_grayscale(image)
    
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, 
        tileGridSize=(tile_size, tile_size)
    )
    enhanced = clahe.apply(image)
    
    return enhanced


def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                 keep_aspect: bool = True) -> np.ndarray:
    """Resize image to target dimensions.
    
    Args:
        target_size: (width, height)
        keep_aspect: If True, pad to maintain aspect ratio
    """
    from PIL import Image

    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode="L")
    else:
        pil_img = Image.fromarray(image)

    if keep_aspect:
        pil_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        # Pad to exact size
        new_img = Image.new(pil_img.mode, target_size, 0)
        offset = (
            (target_size[0] - pil_img.size[0]) // 2,
            (target_size[1] - pil_img.size[1]) // 2,
        )
        new_img.paste(pil_img, offset)
        return np.array(new_img)
    else:
        resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized)


def normalize(image: np.ndarray, mean: float = 0.5, 
              std: float = 0.5) -> np.ndarray:
    """Normalize pixel values to [-1, 1] or [0, 1] range."""
    img_float = image.astype(np.float32) / 255.0
    return (img_float - mean) / std


def normalize_01(image: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] range."""
    img = image.astype(np.float32)
    pmin, pmax = img.min(), img.max()
    if pmax > pmin:
        return (img - pmin) / (pmax - pmin)
    return np.zeros_like(img)


def preprocess_xray(filepath: str, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Full preprocessing pipeline for X-ray images.
    
    Handles: camera photos, DICOM, JPEG, PNG
    Steps: load → grayscale → CLAHE → resize → normalize
    """
    image = load_image(filepath)
    gray = to_grayscale(image)
    enhanced = apply_clahe(gray)
    resized = resize_image(enhanced, target_size, keep_aspect=True)
    normalized = normalize_01(resized)
    return normalized


def preprocess_photo(filepath: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess a camera/phone photo (e.g., oral cancer, ECG strip).
    
    Returns RGB normalized array.
    """
    image = load_image(filepath)
    resized = resize_image(image, target_size, keep_aspect=True)
    normalized = resized.astype(np.float32) / 255.0
    return normalized
