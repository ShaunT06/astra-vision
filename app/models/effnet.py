"""EfficientNet-B0 (timm, Apache 2.0): a small ImageNet-pretrained CNN used as a
second, architecturally different feature extractor for the model comparison."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import config
from .base import DEVICE, Backbone


class EffNetBackbone(Backbone):
    name = "efficientnet_b0"

    def __init__(self, model_id: str = config.EFFNET_MODEL_ID):
        super().__init__()
        import timm

        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).eval().to(DEVICE)
        self.data_cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**self.data_cfg)  # resize, centre-crop, normalise
        self.dim = self.model.num_features

    def preprocess(self, images):
        return torch.stack([self.transform(im) for im in images]).to(DEVICE)

    def features(self, pixel_values):
        return F.normalize(self.model(pixel_values), dim=-1)

    def mean_std(self):
        return list(self.data_cfg["mean"]), list(self.data_cfg["std"])

    def cam_target_layer(self):
        return self.model.conv_head
