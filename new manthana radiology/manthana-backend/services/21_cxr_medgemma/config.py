"""Configuration for CXR MedGemma service."""
import os

SERVICE_NAME = "cxr_medgemma"
PORT = int(os.getenv("PORT", "8019"))
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cuda")

# MedGemma model id (informational; shared/medical_document_parser pins google/medgemma-4b-it)
MEDGEMMA_MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
MEDGEMMA_VRAM_GB = 9.0

SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "5"))
MIN_QUESTIONS = int(os.getenv("MIN_QUESTIONS", "3"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "5"))

# Kimi final report — SSOT role in config/cloud_inference.yaml
KIMI_REPORT_ROLE = os.getenv("KIMI_REPORT_ROLE", "narrative_cxr_medgemma_final")
