import os

SERVICE_NAME = "abdominal_ct"
PORT = 8008
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
DEVICE = os.getenv("DEVICE", "cuda")

# Sybil lung cancer risk scoring (chest CT)
SYBIL_ENABLED = os.getenv("SYBIL_ENABLED", "true").lower() in ("1", "true", "yes")

# Chest heuristics (TB, NAFLD, tropical pancreatitis)
TB_HEURISTIC_ENABLED = os.getenv("TB_HEURISTIC_ENABLED", "true").lower() in ("1", "true", "yes")
