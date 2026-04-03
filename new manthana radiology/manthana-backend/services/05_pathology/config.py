import os
SERVICE_NAME = "pathology"; PORT = 8005
MODEL_DIR = os.getenv("MODEL_DIR", "/models"); DEVICE = os.getenv("DEVICE", "cuda")
