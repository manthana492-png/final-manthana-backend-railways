import os
SERVICE_NAME = "abdominal_ct"; PORT = 8008
MODEL_DIR = os.getenv("MODEL_DIR", "/models"); DEVICE = os.getenv("DEVICE", "cuda")
