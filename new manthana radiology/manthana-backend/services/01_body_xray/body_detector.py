"""
Manthana — X-Ray Body Region Detector
Uses torchxrayvision (shared singleton) to infer body region from pathology activations.
"""

import sys
import logging
import numpy as np

sys.path.insert(0, "/app/shared")

from preprocessing.image_utils import load_image, to_grayscale
from txrv_utils import triage_forward_probs

logger = logging.getLogger("manthana.body_detector")

def detect_body_region(filepath: str) -> str:
    """Auto-detect which body region an X-ray shows.

    Uses torchxrayvision DenseNet (shared with triage/chest) on canonical preprocess.
    Falls back to image aspect ratio + intensity heuristics.

    Returns: body region string (chest, extremity, spine, abdomen, skull, pelvis)
    """
    try:
        return _detect_with_torchxrayvision(filepath)
    except Exception as e:
        logger.warning(f"torchxrayvision detection failed: {e}. Using heuristic.")
        return _detect_with_heuristic(filepath)


def _detect_with_torchxrayvision(filepath: str) -> str:
    """Use shared TXRV all-weights forward for pathology-based region hint."""
    probs, names = triage_forward_probs(filepath)
    pathologies = {names[i]: float(probs[i]) for i in range(len(names))}

    chest_indicators = sum(
        [
            pathologies.get("Cardiomegaly", 0),
            pathologies.get("Lung Opacity", 0),
            pathologies.get("Effusion", 0),
            pathologies.get("Atelectasis", 0),
        ]
    )

    if chest_indicators > 0.5:
        return "chest"

    return _detect_with_heuristic(filepath)


def _detect_with_heuristic(filepath: str) -> str:
    """Heuristic body region detection based on image properties.

    Uses aspect ratio, intensity distribution, and edge patterns.
    """
    image = load_image(filepath)
    gray = to_grayscale(image)
    h, w = gray.shape[:2]

    aspect = w / h

    # Intensity features
    mean_intensity = np.mean(gray)

    # Region-based intensity analysis
    top_half = np.mean(gray[: h // 2, :])
    bottom_half = np.mean(gray[h // 2 :, :])
    left_half = np.mean(gray[:, : w // 2])
    right_half = np.mean(gray[:, w // 2 :])

    # Heuristic rules based on typical X-ray characteristics

    # Chest: roughly square, symmetric, dark lung fields at top
    if 0.7 < aspect < 1.4 and top_half < bottom_half:
        return "chest"

    # Spine: tall and narrow
    if aspect < 0.6:
        return "spine"

    # Extremity: variable aspect, usually lighter overall
    if aspect > 1.5 or aspect < 0.5:
        return "extremity"

    # Pelvis: wide, lower body
    if aspect > 1.2 and mean_intensity > 100:
        return "pelvis"

    # Abdomen: similar to chest but different intensity pattern
    if 0.7 < aspect < 1.3 and mean_intensity > 80:
        return "abdomen"

    # Default: chest (most common X-ray type)
    return "chest"
