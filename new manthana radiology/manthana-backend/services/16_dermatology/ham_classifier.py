"""
HAM10000-style 7-class EfficientNet-V2-M classifier.

Checkpoint: state_dict for torchvision efficientnet_v2_m with final linear out_features=7.
Class order must match config.HAM7_CLASS_ORDER.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("manthana.dermatology.ham_classifier")


class DermHamClassifier:
    def __init__(self, model_path: Path, class_keys: tuple[str, ...], *, device: str = "cpu"):
        self.device = device
        self.class_keys = class_keys
        if len(class_keys) != 7:
            raise ValueError("HAM classifier expects 7 class keys")
        self.model = self._build_model()
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _build_model(self) -> nn.Module:
        from torchvision.models import efficientnet_v2_m

        m = efficientnet_v2_m(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 7),
        )
        return m

    def classify(self, pil_image: Image.Image) -> dict[str, float]:
        t = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        return {k: round(float(p), 4) for k, p in zip(self.class_keys, probs)}

    def model_for_cam(self) -> nn.Module:
        return self.model
