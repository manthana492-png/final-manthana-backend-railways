"""Manthana — Pathology Preprocessing"""
import sys; sys.path.insert(0, "/app/shared")
from preprocessing.wsi_utils import extract_tiles, get_slide_info
from preprocessing.image_utils import load_image, resize_image
