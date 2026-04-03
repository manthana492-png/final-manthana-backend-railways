"""
Manthana — Lab Report config
"""

import os

SERVICE_NAME = "lab_report"
PORT = 8015

KIMI_API_KEY = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")
KIMI_LAB_MODEL = os.getenv("KIMI_LAB_MODEL") or KIMI_MODEL
