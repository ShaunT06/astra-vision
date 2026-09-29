"""Model registry.

* siglip-zeroshot : SigLIP 2 image encoder, head built from *text prompts* (no training).
* siglip-probe    : SigLIP 2 image encoder, head = logistic regression trained on our images.
* effnet-probe    : EfficientNet-B0 (ImageNet CNN), head = logistic regression on our images.
"""
from __future__ import annotations

import json
from functools import lru_cache

from .. import config
from .base import Backbone, Classifier, LinearHead

MODEL_INFO = {
    "siglip-probe": "SigLIP 2 image encoder + logistic-regression head trained on ASTRA images (few-shot transfer learning).",
    "siglip-zeroshot": "SigLIP 2 zero-shot: compares the image with text descriptions of each class. No training.",
    "effnet-probe": "EfficientNet-B0 CNN (ImageNet) + logistic-regression head trained on ASTRA images.",
}
TRAINED = ("siglip-probe", "effnet-probe")


@lru_cache(maxsize=None)
def get_backbone(name: str) -> Backbone:
    if name == "siglip2":
        from .siglip import SiglipBackbone
        return SiglipBackbone()
    from .effnet import EffNetBackbone
    return EffNetBackbone()


def backbone_for(key: str) -> str:
    return "efficientnet_b0" if key.startswith("effnet") else "siglip2"


def zeroshot_head() -> LinearHead:
    prompts = {config.CLASSES[c]: config.CLASS_PROMPTS[c] for c in config.CLASS_IDS}
    return get_backbone("siglip2").prompt_head(prompts)


@lru_cache(maxsize=None)
def get_classifier(key: str) -> Classifier:
    if key not in MODEL_INFO:
        raise KeyError(key)
    if key == "siglip-zeroshot":
        head = zeroshot_head()
    else:
        path = config.ARTIFACTS_DIR / f"{key}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m scripts.train` first.")
        head = LinearHead.load(path)
    return Classifier(key, get_backbone(backbone_for(key)), head, MODEL_INFO[key])


def available_models() -> list[str]:
    return ["siglip-zeroshot"] + [k for k in TRAINED if (config.ARTIFACTS_DIR / f"{k}.npz").exists()]


@lru_cache(maxsize=1)
def get_ood_head() -> LinearHead:
    """Zero-shot head over defence classes + non-defence concepts, for the OOD gate."""
    groups = {f"defence:{c}": config.CLASS_PROMPTS[c] for c in config.CLASS_IDS}
    groups.update({f"other:{k}": v for k, v in config.NON_DEFENCE_PROMPTS.items()})
    return get_backbone("siglip2").prompt_head(groups)


def load_metrics() -> dict:
    path = config.REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}
