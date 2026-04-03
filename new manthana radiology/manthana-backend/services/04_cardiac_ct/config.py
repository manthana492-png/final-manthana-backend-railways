import os
SERVICE_NAME = "cardiac_ct"; PORT = 8004
MODEL_DIR = os.getenv("MODEL_DIR", "/models"); DEVICE = os.getenv("DEVICE", "cuda")
