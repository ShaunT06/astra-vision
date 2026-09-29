"""Multi-object detection with bounding boxes.

OWLv2 (Google, Apache 2.0) is an *open-vocabulary* detector: it finds boxes for
free-text queries, so no box-labelled training data is needed. Each box is then
cropped and classified by our own classifier, which assigns the final label and
confidence. Two-stage = detector for "where", classifier for "what".
"""
from __future__ import annotations

import time
from functools import lru_cache

import torch
from PIL import Image, ImageDraw, ImageFont

from . import config
from .inference import to_data_uri
from .models import get_classifier

QUERIES = [
    "a military aircraft", "a fighter jet", "a helicopter", "a drone", "a tank",
    "an armored vehicle", "a military truck", "a warship", "a ship", "a submarine",
]
MAX_BOXES = 10
COLOURS = ["#E0B03A", "#4FC3F7", "#81C784", "#FF8A65", "#BA68C8", "#F06292"]


@lru_cache(maxsize=1)
def _owl():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    return (Owlv2Processor.from_pretrained(config.OWL_MODEL_ID),
            Owlv2ForObjectDetection.from_pretrained(config.OWL_MODEL_ID).eval())


def _boxes(img: Image.Image, min_score: float):
    processor, model = _owl()
    inputs = processor(text=[QUERIES], images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # OWLv2 pads the image to a square before resizing, so boxes are relative to the padded square.
    side = max(img.size)
    post = getattr(processor, "post_process_grounded_object_detection", None) or processor.post_process_object_detection
    res = post(outputs=outputs, threshold=min_score, target_sizes=torch.tensor([[side, side]]))[0]
    boxes, scores, labels = res["boxes"], res["scores"], res["labels"]
    if len(boxes) == 0:
        return []
    from torchvision.ops import nms

    keep = nms(boxes, scores, iou_threshold=0.4).tolist()   # class-agnostic: one box per object
    W, H = img.size
    keep = [i for i in keep if not _is_group_box(boxes[i], [boxes[j] for j in keep if j != i], W * H)][:MAX_BOXES]
    out = []
    for i in keep:
        x0, y0, x1, y1 = boxes[i].tolist()
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        out.append({"box": [round(x0), round(y0), round(x1), round(y1)], "detector_score": round(float(scores[i]), 3),
                    "detector_query": QUERIES[int(labels[i])]})
    return out


def _is_group_box(box, others, image_area: float, inside: float = 0.8) -> bool:
    """NMS keeps a big box around several objects (IoU with each is small). Drop a *large*
    box (>= 25% of the image) that mostly contains two or more other detections - it
    describes a group or a whole scene, not an object. Small boxes are never dropped, so a
    near helicopter with two distant ones inside its box survives."""
    def area(b):
        return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

    if area(box) < 0.25 * image_area:
        return False
    contained = 0
    for o in others:
        ix = max(0.0, float(min(box[2], o[2]) - max(box[0], o[0])))
        iy = max(0.0, float(min(box[3], o[3]) - max(box[1], o[1])))
        if area(o) > 0 and ix * iy / area(o) >= inside and area(o) < area(box):
            contained += 1
    return contained >= 2


def _crop(img: Image.Image, box, pad: float = 0.1) -> Image.Image:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((max(0, x0 - dx), max(0, y0 - dy), min(img.width, x1 + dx), min(img.height, y1 + dy)))


def _draw(img: Image.Image, dets: list[dict]) -> Image.Image:
    im = img.copy()
    scale = 1024 / max(im.size) if max(im.size) > 1024 else 1.0
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, im.width // 45))
    except OSError:
        font = ImageFont.load_default()
    for n, det in enumerate(dets):
        c = COLOURS[n % len(COLOURS)]
        x0, y0, x1, y1 = (v * scale for v in det["box"])
        d.rectangle((x0, y0, x1, y1), outline=c, width=max(2, im.width // 300))
        text = f"{det['label']} {det['confidence']:.0%}"
        tb = d.textbbox((x0, y0), text, font=font)
        ty = y0 - (tb[3] - tb[1]) - 6 if y0 > (tb[3] - tb[1]) + 6 else y0
        d.rectangle((x0, ty, x0 + tb[2] - tb[0] + 8, ty + tb[3] - tb[1] + 6), fill=c)
        d.text((x0 + 4, ty + 1), text, fill="#111111", font=font)
    return im


def detect_objects(img: Image.Image, model_key: str, min_score: float = 0.15) -> dict:
    t0 = time.perf_counter()
    clf = get_classifier(model_key)
    dets = _boxes(img, min_score)
    if dets:
        probs = clf.probs([_crop(img, d["box"]) for d in dets])
        for d, p in zip(dets, probs):
            k = int(p.argmax())
            d["label"] = clf.labels[k]
            d["confidence"] = round(float(p[k]), 4)
            d["uncertain"] = bool(p[k] < clf.thresholds["min_confidence"])
    counts: dict[str, int] = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return {
        "model": model_key,
        "detector": config.OWL_MODEL_ID,
        "count": len(dets),
        "counts": counts,
        "detections": dets,
        "annotated_image": to_data_uri(_draw(img, dets)),
        "image_size": list(img.size),
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }
