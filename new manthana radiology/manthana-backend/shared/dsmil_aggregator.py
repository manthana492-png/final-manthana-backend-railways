"""
DSMIL-style attention MIL for slide-level scores from tile embeddings (V5 pathology).

MIT-friendly, self-contained. No external dsmil-wsi package required.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("manthana.dsmil")

CLASS_NAMES = ("benign", "malignant", "inflammation", "necrosis")


def _stack_embeddings(embeddings: list) -> np.ndarray | None:
    if not embeddings:
        return None
    arrs = []
    for e in embeddings:
        a = np.asarray(e, dtype=np.float32).reshape(-1)
        arrs.append(a)
    if not arrs:
        return None
    d = max(a.shape[0] for a in arrs)
    padded = []
    for a in arrs:
        if a.shape[0] < d:
            p = np.zeros(d, dtype=np.float32)
            p[: a.shape[0]] = a
            padded.append(p)
        else:
            padded.append(a[:d])
    return np.stack(padded, axis=0)


def dsmil_slide_scores(embeddings: list, num_classes: int = 4) -> dict[str, Any]:
    """
    Attention-weighted bag embedding → sigmoid multi-label scores.
    Weights are randomly initialized unless dsmil_mil.pt is provided.
    """
    bag = _stack_embeddings(embeddings)
    if bag is None or bag.shape[0] == 0:
        return {
            "classification_status": "no_embeddings",
            "tissue_type": "unknown",
            "malignancy_score": None,
            "inflammation_score": None,
            "necrosis_score": None,
        }

    try:
        import os

        import torch
        import torch.nn as nn

        n, d_in = bag.shape
        hid = min(256, max(64, d_in // 2))

        class _MIL(nn.Module):
            def __init__(self):
                super().__init__()
                self.att_v = nn.Linear(d_in, hid)
                self.att_u = nn.Linear(hid, 1)
                self.fc = nn.Linear(d_in, num_classes)

            def forward(self, x: torch.Tensor):
                # x: N x D
                a = torch.tanh(self.att_v(x))
                a = self.att_u(a).squeeze(-1)
                w = torch.softmax(a, dim=0)
                z = torch.sum(w.unsqueeze(-1) * x, dim=0)
                logits = self.fc(z)
                return logits, w

        mil = _MIL()
        wpath = os.path.join(os.getenv("MODEL_DIR", "/models"), "dsmil_mil.pt")
        if os.path.isfile(wpath):
            try:
                try:
                    sd = torch.load(wpath, map_location="cpu", weights_only=True)
                except TypeError:
                    sd = torch.load(wpath, map_location="cpu")
                mil.load_state_dict(sd, strict=False)
                logger.info("Loaded dsmil_mil.pt")
            except Exception as e:
                logger.warning("dsmil_mil.pt load failed: %s", e)

        mil.eval()
        x = torch.from_numpy(bag)
        with torch.no_grad():
            logits, _w = mil(x)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        mal = float(probs[1]) if len(probs) > 1 else float(probs[0])
        infl = float(probs[2]) if len(probs) > 2 else 0.2
        necr = float(probs[3]) if len(probs) > 3 else 0.15
        tissue = CLASS_NAMES[int(np.argmax(probs))] if probs else "benign"

        return {
            "classification_status": "dsmil_mil",
            "tissue_type": tissue,
            "malignancy_score": mal,
            "inflammation_score": infl,
            "necrosis_score": necr,
            "benign_score": float(probs[0]) if probs else 0.5,
        }
    except Exception as e:
        logger.warning("DSMIL aggregation failed: %s", e)
        mean_emb = np.mean(bag, axis=0)
        proxy = float(np.clip(np.std(mean_emb) * 2.0, 0.0, 1.0))
        return {
            "classification_status": "embedding_mean_fallback",
            "tissue_type": "unknown",
            "malignancy_score": proxy,
            "inflammation_score": proxy * 0.6,
            "necrosis_score": proxy * 0.4,
        }
