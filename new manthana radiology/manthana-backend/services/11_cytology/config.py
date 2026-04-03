import os
SERVICE_NAME = "cytology"; PORT = 8011
MODEL_DIR = os.getenv("MODEL_DIR", "/models"); DEVICE = os.getenv("DEVICE", "cuda")
