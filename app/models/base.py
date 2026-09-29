"""Shared building blocks: every classifier is `frozen backbone -> linear head`.

With only ~150 labelled images, training a whole network would overfit; a linear
head on strong frozen features is the standard few-shot recipe. Because every
model ends in a plain `nn.Linear`, the same Grad-CAM code explains all of them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .. import config

# Grad mode is thread-local (FastAPI runs requests in worker threads), so every
# inference entry point is decorated with @torch.no_grad() instead of a global switch.
DEVICE = "cpu"


def as_tensor(out) -> torch.Tensor:
    # transformers has changed get_*_features return types across versions.
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


class Backbone(nn.Module):
    """Frozen feature extractor: PIL images -> L2-normalised embeddings."""

    name: str
    dim: int

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            px = self.preprocess(images[i : i + batch_size])
            out.append(self.features(px).cpu().numpy())
        return np.concatenate(out)

    # Grad-CAM plumbing
    cam_eigen_smooth = False

    def cam_target_layer(self) -> nn.Module:
        raise NotImplementedError

    def cam_reshape(self):
        return None

    def mean_std(self) -> tuple[list[float], list[float]]:
        raise NotImplementedError


class LinearHead(nn.Module):
    def __init__(self, W: torch.Tensor, b: torch.Tensor, labels: list[str]):
        super().__init__()
        self.linear = nn.Linear(W.shape[1], W.shape[0])
        self.linear.weight.data = W.float().clone()
        self.linear.bias.data = b.float().clone()
        self.linear.requires_grad_(False)   # frozen; Grad-CAM still gets gradients via the backbone
        self.labels = labels

    def forward(self, x):
        return self.linear(x)

    def save(self, path):
        np.savez(path, W=self.linear.weight.numpy(), b=self.linear.bias.numpy(), labels=np.array(self.labels))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(torch.from_numpy(d["W"]), torch.from_numpy(d["b"]), [str(x) for x in d["labels"]])

    @classmethod
    def from_sklearn(cls, clf, labels):
        return cls(torch.from_numpy(clf.coef_), torch.from_numpy(clf.intercept_), labels)


class Classifier(nn.Module):
    """backbone + head as a single differentiable module (needed for Grad-CAM)."""

    def __init__(self, key: str, backbone: Backbone, head: LinearHead, description: str):
        super().__init__()
        self.key = key
        self.backbone = backbone
        self.head = head
        self.description = description
        self.thresholds = config.load_thresholds(key)

    @property
    def labels(self):
        return self.head.labels

    def forward(self, pixel_values):
        return self.head(self.backbone.features(pixel_values))

    @torch.no_grad()
    def logits(self, images: list[Image.Image]) -> torch.Tensor:
        return self.head(torch.from_numpy(self.backbone.embed(images)))

    def probs(self, images: list[Image.Image]) -> np.ndarray:
        T = self.thresholds.get("temperature", 1.0)       # temperature scaling (calibration)
        return torch.softmax(self.logits(images) / T, dim=-1).numpy()
