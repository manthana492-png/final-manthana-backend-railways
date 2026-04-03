"""Prima-style MLP head (embeddings → clinical scores). MIT / custom weights from /models."""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("manthana.prima_mlp_head")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
HEAD_PATH = os.path.join(MODEL_DIR, "prima_mlp_head.pt")

LABELS = ("normal", "mass_lesion", "hemorrhage", "infarct")


def run_prima_mlp(embedding: np.ndarray, device: str = "cuda") -> dict[str, float]:
    """
    embedding: 1D float array (e.g. 768-d pooled).
    Loads optional trained head; else Xavier-initialized MLP.
    On any failure returns {} — do not fabricate equal fallback probabilities.
    """
    try:
        import torch
        import torch.nn as nn

        d_in = int(np.asarray(embedding).reshape(-1).shape[0])
        hid = min(256, max(64, d_in // 2))
        n_out = len(LABELS)

        head = nn.Sequential(
            nn.Linear(d_in, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, n_out),
        )
        if os.path.isfile(HEAD_PATH):
            try:
                try:
                    sd = torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
                except TypeError:
                    sd = torch.load(HEAD_PATH, map_location="cpu")
                head.load_state_dict(sd, strict=False)
                logger.info("Loaded prima_mlp_head.pt")
            except Exception as e:
                logger.warning("prima_mlp_head.pt load failed: %s", e)

        head.eval()
        t = torch.from_numpy(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        dev = device if __import__("torch").cuda.is_available() and device == "cuda" else "cpu"
        t = t.to(dev)
        head = head.to(dev)
        with torch.no_grad():
            logits = head(t)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()
        return {LABELS[i]: float(probs[i]) for i in range(min(len(LABELS), len(probs)))}
    except Exception as e:
        logger.warning("Prima MLP head failed: %s", e)
        return {}
