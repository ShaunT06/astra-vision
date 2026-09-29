"""SigLIP 2 (Google, Apache 2.0): a vision-language encoder. Used three ways:
zero-shot classifier (text-prompt head), feature extractor for the trained probe,
and the out-of-distribution gate."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import config
from .base import DEVICE, Backbone, LinearHead, as_tensor


class SiglipBackbone(Backbone):
    name = "siglip2"

    def __init__(self, model_id: str = config.SIGLIP_MODEL_ID):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(DEVICE)
        vc = self.model.config.vision_config
        self.dim = vc.hidden_size
        self.grid = vc.image_size // vc.patch_size
        self.logit_scale = self.model.logit_scale.exp().item()
        self.logit_bias = self.model.logit_bias.item()

    def preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)

    def features(self, pixel_values):
        f = as_tensor(self.model.get_image_features(pixel_values=pixel_values))
        return F.normalize(f, dim=-1)

    @torch.no_grad()
    def text_embed(self, prompts: list[str]) -> torch.Tensor:
        tok = self.processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
        f = as_tensor(self.model.get_text_features(input_ids=tok["input_ids"].to(DEVICE)))
        return F.normalize(f, dim=-1)

    @torch.no_grad()
    def prompt_head(self, prompt_groups: dict[str, list[str]]) -> LinearHead:
        """One weight vector per class = mean of its prompt embeddings (prompt ensembling)."""
        rows = [F.normalize(self.text_embed(p).mean(0), dim=-1) for p in prompt_groups.values()]
        W = torch.stack(rows) * self.logit_scale
        b = torch.full((len(rows),), self.logit_bias)
        return LinearHead(W, b, list(prompt_groups))

    def mean_std(self):
        ip = self.processor.image_processor
        return list(ip.image_mean), list(ip.image_std)

    # Plain Grad-CAM on ViTs lights up a few high-norm background tokens ("Vision Transformers
    # Need Registers", Darcet et al. 2023). Keeping only the first principal component of the
    # gradient-weighted activations (eigen-smooth) suppresses them - compared visually in reports/.
    cam_eigen_smooth = True

    def cam_target_layer(self):
        return self.model.vision_model.encoder.layers[-1].layer_norm1

    def cam_reshape(self):
        g = self.grid

        def reshape(t):  # (B, tokens, C) -> (B, C, g, g); SigLIP has no CLS token
            t = t[:, -g * g :, :]
            return t.reshape(t.size(0), g, g, t.size(2)).permute(0, 3, 1, 2)

        return reshape
