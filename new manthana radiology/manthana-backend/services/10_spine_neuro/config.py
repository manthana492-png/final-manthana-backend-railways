import os
SERVICE_NAME = "spine_neuro"; PORT = 8010
MODEL_DIR = os.getenv("MODEL_DIR", "/models"); DEVICE = os.getenv("DEVICE", "cuda")
