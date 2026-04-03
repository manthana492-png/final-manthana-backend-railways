import os
SERVICE_NAME = "brain_mri"
PORT = 8002
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cuda")
