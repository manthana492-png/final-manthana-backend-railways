import os
SERVICE_NAME = "ultrasound"; PORT = 8009
MODEL_DIR = os.getenv("MODEL_DIR", "/models"); DEVICE = os.getenv("DEVICE", "cuda")
