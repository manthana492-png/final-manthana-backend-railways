import os
SERVICE_NAME = "mammography"; PORT = 8012
MODEL_DIR = os.getenv("MODEL_DIR", "/models"); DEVICE = os.getenv("DEVICE", "cuda")
