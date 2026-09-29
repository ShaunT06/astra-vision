"""Export the SigLIP 2 image encoder + all heads for the serverless (Vercel) deployment.

PyTorch is far too large for a Vercel function, so the deployed API uses ONNX Runtime:
  web/api/model/siglip_vision.onnx   weight-only int8 vision tower -> (image_embeds, patch_tokens)
  web/api/model/heads.npz            trained probe head, zero-shot OOD head, thresholds

Only the weights are int8 (per-output-channel, dequantised to fp32 at run time). Dynamic int8, which also
quantises activations, was tried first and ruined SigLIP's embeddings (cosine 0.67 vs PyTorch) because
of its large activation outliers; weight-only keeps cosine at 0.999.

Run (needs `pip install onnx onnxruntime onnxscript`):  python -m scripts.export_onnx
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app import config
from app.models import get_backbone, get_ood_head, get_classifier
from app.models.base import as_tensor

OUT = config.ROOT / "web" / "api" / "model"


def weight_only_int8(src: Path, dst: Path, min_elems: int = 100_000) -> int:
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    dequant, drop = [], set()
    for node in m.graph.node:
        if node.op_type == "MatMul" and node.input[1] in inits:
            w = numpy_helper.to_array(inits[node.input[1]])
            if w.ndim == 2 and w.size >= min_elems:
                scale = np.maximum(np.abs(w).max(axis=0), 1e-8) / 127.0
                q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
                base = node.input[1]
                m.graph.initializer.extend([
                    numpy_helper.from_array(q, base + "_q"),
                    numpy_helper.from_array(scale.astype(np.float32), base + "_s"),
                    numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), base + "_z"),
                ])
                dequant.append(helper.make_node("DequantizeLinear", [base + "_q", base + "_s", base + "_z"],
                                                [base + "_dq"], axis=1, name=base + "_dq_node"))
                node.input[1] = base + "_dq"
                drop.add(base)
    keep = [i for i in m.graph.initializer if i.name not in drop]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = dequant + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, str(dst))
    return len(dequant)


class VisionTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, pixel_values):
        out = self.m.vision_model(pixel_values=pixel_values)
        return out.pooler_output, out.last_hidden_state


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bb = get_backbone("siglip2")
    tower = VisionTower(bb.model).eval()
    dummy = torch.randn(1, 3, 224, 224)
    fp32 = OUT / "siglip_vision_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(tower, dummy, str(fp32), input_names=["pixel_values"],
                          output_names=["pooled", "tokens"], opset_version=17, dynamo=False)

    print("int8 weight matrices:", weight_only_int8(fp32, OUT / "siglip_vision.onnx"))
    fp32.unlink()
    for extra in OUT.glob("*.data"):
        extra.unlink()

    probe = get_classifier("siglip-probe").head
    ood = get_ood_head()
    th = config.load_thresholds("siglip-probe")
    np.savez(OUT / "heads.npz",
             probe_W=probe.linear.weight.numpy(), probe_b=probe.linear.bias.numpy(),
             probe_labels=np.array(probe.labels),
             ood_W=ood.linear.weight.numpy(), ood_b=ood.linear.bias.numpy(), ood_labels=np.array(ood.labels),
             thresholds=np.array(json.dumps(th)))
    print("wrote", [f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in OUT.iterdir()])


if __name__ == "__main__":
    main()
