"""
Manthana — Video Utilities
Key frame extraction for ultrasound video processing.
"""

import os
import logging
import tempfile
from typing import List
import numpy as np

logger = logging.getLogger("manthana.video_utils")


def extract_key_frames(video_path: str, max_frames: int = 30,
                       interval_sec: float = None) -> List[np.ndarray]:
    """Extract key frames from an ultrasound video.
    
    Args:
        video_path: Path to video file (.mp4, .avi)
        max_frames: Maximum number of frames to extract
        interval_sec: Extract one frame every N seconds.
                     If None, extracts frames uniformly across video.
    
    Returns:
        List of RGB numpy arrays (frames)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        raise ValueError(f"Invalid video metadata: frames={total_frames}, fps={fps}")

    # Determine which frames to extract
    if interval_sec:
        frame_interval = int(fps * interval_sec)
        frame_indices = list(range(0, total_frames, frame_interval))
    else:
        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))

    frame_indices = frame_indices[:max_frames]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path} "
                f"(total={total_frames}, fps={fps:.1f})")
    return frames


def save_frames(frames: List[np.ndarray], output_dir: str = None) -> List[str]:
    """Save extracted frames as PNG files.
    
    Returns list of file paths.
    """
    from PIL import Image

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="manthana_frames_")
    
    os.makedirs(output_dir, exist_ok=True)
    
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"frame_{i:04d}.png")
        Image.fromarray(frame).save(path)
        paths.append(path)
    
    return paths
