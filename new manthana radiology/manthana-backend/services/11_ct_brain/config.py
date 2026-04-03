import os

SERVICE_NAME = "ct_brain"
PORT = int(os.getenv("PORT", "8017"))
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
