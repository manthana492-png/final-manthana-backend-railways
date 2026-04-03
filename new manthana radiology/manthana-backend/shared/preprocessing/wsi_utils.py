"""
Manthana — WSI (Whole Slide Image) Utilities
Tile extraction for pathology and cytology processing.
"""

import logging
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

logger = logging.getLogger("manthana.wsi_utils")


def _open_slide(path: str) -> tuple[Any | None, str]:
    """
    Try openslide, then tifffile, then Pillow large image.
    Never raises — returns (None, error) on total failure.
    """
    try:
        import openslide

        return openslide.OpenSlide(str(path)), "openslide"
    except ImportError:
        pass
    except Exception as e:
        logger.debug("openslide failed: %s", e)
    try:
        import tifffile

        return tifffile.TiffFile(str(path)), "tifffile"
    except Exception as e:
        logger.debug("tifffile open failed: %s", e)
    try:
        from PIL import Image

        img = Image.open(str(path))
        return img, "pillow"
    except Exception as e:
        return None, f"failed: {e}"


def _tiles_from_array(
    arr: np.ndarray,
    tile_size: int,
    overlap: int,
    tissue_threshold: float,
) -> List[Tuple[np.ndarray, dict]]:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3].copy()
    h, w = arr.shape[:2]
    step = tile_size - overlap
    tiles: list = []
    for y in range(0, max(1, h - tile_size + 1), step):
        for x in range(0, max(1, w - tile_size + 1), step):
            patch = arr[y : y + tile_size, x : x + tile_size]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                continue
            if _is_tissue(patch, tissue_threshold):
                tiles.append(
                    (
                        patch,
                        {"x": x, "y": y, "level": 0, "tile_size": tile_size},
                    )
                )
    return tiles


def extract_tiles(
    wsi_path: str,
    tile_size: int = 256,
    overlap: int = 0,
    level: int = 0,
    tissue_threshold: float = 0.5,
) -> List[Tuple[np.ndarray, dict]]:
    """Extract tissue tiles from a whole slide image.

    Filters out background tiles (mostly white/empty).
    Uses OpenSlide when available; otherwise tifffile or Pillow.
    """
    slide, backend = _open_slide(wsi_path)
    if slide is None:
        logger.warning("Could not open WSI: %s (%s)", wsi_path, backend)
        return []

    tiles: List[Tuple[np.ndarray, dict]] = []

    try:
        if backend == "openslide":
            dims = slide.level_dimensions[level]
            logger.info(
                "WSI opened (openslide): %s, dims=%s, levels=%s",
                wsi_path,
                dims,
                slide.level_count,
            )
            step = tile_size - overlap
            for y in range(0, dims[1], step):
                for x in range(0, dims[0], step):
                    tile = slide.read_region(
                        location=(x, y),
                        level=level,
                        size=(tile_size, tile_size),
                    )
                    tile_array = np.array(tile.convert("RGB"))
                    if _is_tissue(tile_array, tissue_threshold):
                        location = {
                            "x": x,
                            "y": y,
                            "level": level,
                            "tile_size": tile_size,
                        }
                        tiles.append((tile_array, location))
            slide.close()

        elif backend == "tifffile":
            with slide:
                page = slide.pages[0]
                arr = page.asarray()
            tiles = _tiles_from_array(arr, tile_size, overlap, tissue_threshold)
            logger.info(
                "WSI via tifffile: %s, extracted %s tiles", wsi_path, len(tiles)
            )

        elif backend == "pillow":
            img = slide
            img.load()
            arr = np.array(img.convert("RGB"))
            tiles = _tiles_from_array(arr, tile_size, overlap, tissue_threshold)
            logger.info(
                "WSI via Pillow: %s, extracted %s tiles", wsi_path, len(tiles)
            )
    except Exception as e:
        logger.warning("Tile extraction failed (%s): %s", backend, e)
        if backend == "openslide" and slide is not None:
            try:
                slide.close()
            except Exception:
                pass
        return []

    logger.info("Extracted %s tissue tiles from %s", len(tiles), wsi_path)
    return tiles


def _is_tissue(tile: np.ndarray, threshold: float = 0.5) -> bool:
    """Check if a tile contains enough tissue (not background)."""
    import cv2

    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    tissue_pixels = np.sum(saturation > 20)
    total_pixels = saturation.size
    return (tissue_pixels / total_pixels) > threshold


def get_slide_info(wsi_path: str) -> dict:
    """Get metadata from a WSI file."""
    slide, backend = _open_slide(wsi_path)
    if slide is None:
        return {"error": backend, "path": wsi_path}
    if backend == "openslide":
        try:
            info = {
                "dimensions": slide.dimensions,
                "level_count": slide.level_count,
                "level_dimensions": [
                    slide.level_dimensions[i] for i in range(slide.level_count)
                ],
                "properties": dict(slide.properties),
                "backend": "openslide",
            }
            slide.close()
            return info
        except Exception as e:
            return {"error": str(e), "path": wsi_path}
    if backend == "tifffile":
        try:
            with slide as tf:
                page = tf.pages[0]
                shape = page.shape
            return {
                "dimensions": shape,
                "level_count": 1,
                "level_dimensions": [shape],
                "backend": "tifffile",
            }
        except Exception as e:
            return {"error": str(e), "path": wsi_path}
    try:
        w, h = slide.size
        return {
            "dimensions": (w, h),
            "level_count": 1,
            "backend": "pillow",
        }
    except Exception as e:
        return {"error": str(e), "path": wsi_path}
