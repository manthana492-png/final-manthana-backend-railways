"""
V2 EfficientNet-B4 classifier — loaded only when checkpoint exists (see analyzer._try_load_classifier).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger("manthana.dermatology.classifier")

DERM_CLASSES = [
    "tinea",
    "vitiligo",
    "psoriasis",
    "melasma",
    "acne",
    "eczema_dermatitis",
    "scabies",
    "urticaria",
    "bcc",
    "scc",
    "melanoma",
    "normal_benign",
]


class DermClassifier:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = device
        self.model = self._build_model()
        state = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self) -> nn.Module:
        m = models.efficientnet_b4(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, len(DERM_CLASSES)),
        )
        return m

    def model_for_cam(self) -> nn.Module:
        return self.model

    def classify(self, pil_image: Image.Image) -> dict:
        t = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        scores = {c: round(p, 4) for c, p in zip(DERM_CLASSES, probs)}
        top = max(scores, key=scores.get)
        conf = scores[top]
        return {
            **scores,
            "top_class": top,
            "confidence": conf,
            "confidence_label": "high"
            if conf >= 0.70
            else "medium"
            if conf >= 0.45
            else "low",
            "is_malignant_candidate": top in {"bcc", "scc", "melanoma"},
        }
