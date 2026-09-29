"""Turns model outputs into the result a user sees: top-3, confidence, uncertainty
flag, out-of-distribution check, plain-English explanation and Grad-CAM heatmap."""
from __future__ import annotations

import base64
import io
import time

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from . import config
from .models import Classifier, get_backbone, get_classifier, get_ood_head

HINTS = {
    "Military Aircraft": "fixed wings, jet or propeller engines and a streamlined fuselage",
    "Military Helicopter": "a main rotor, tail boom and cabin-shaped fuselage",
    "Drone / UAV": "a small unmanned airframe, often with multiple rotors or no cockpit",
    "Military Land Vehicle": "tracks or large wheels, armour plating and often a turret",
    "Naval Vessel": "a hull on water, superstructure and deck equipment",
}


def classify(img: Image.Image, model_key: str = config.DEFAULT_MODEL, explain: bool = True) -> dict:
    t0 = time.perf_counter()
    clf = get_classifier(model_key)
    probs = clf.probs([img])[0]
    order = np.argsort(probs)[::-1]
    top3 = [{"label": clf.labels[i], "confidence": round(float(probs[i]), 4)} for i in order[:3]]
    best, second = probs[order[0]], probs[order[1]]

    th = clf.thresholds
    reasons = []
    if best < th["min_confidence"]:
        reasons.append(f"top confidence {best:.0%} is below the {th['min_confidence']:.0%} threshold")
    if best - second < th["min_margin"]:
        reasons.append(f"the top two classes are only {best - second:.0%} apart")

    ood = out_of_distribution(img)
    recognised = not ood["is_ood"]

    result = {
        "model": model_key,
        "prediction": top3[0]["label"] if recognised else None,
        "confidence": top3[0]["confidence"],
        "top3": top3,
        "uncertain": bool(reasons) or not recognised,
        "uncertainty_reasons": reasons,
        "defence_object_detected": recognised,
        "ood": ood,
        "explanation": explanation(top3, reasons, ood),
        "image_size": list(img.size),
    }
    if explain:
        result["heatmap"] = gradcam_png(clf, img, int(order[0]))
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


@torch.no_grad()
def out_of_distribution(img: Image.Image) -> dict:
    """Zero-shot check: does the image look more like a non-defence concept than any
    defence class? Softmax classifiers always pick *some* class, so without this a cat
    photo would be reported as e.g. 'Drone 91%'."""
    head = get_ood_head()
    feats = torch.from_numpy(get_backbone("siglip2").embed([img]))
    p = torch.softmax(head(feats), dim=-1)[0].numpy()
    defence = sum(p[i] for i, l in enumerate(head.labels) if l.startswith("defence:"))
    j = int(np.argmax(p))
    top = head.labels[j]
    return {
        "is_ood": top.startswith("other:"),
        "defence_score": round(float(defence), 4),
        "closest_concept": top.split(":", 1)[1],
    }


def explanation(top3: list[dict], reasons: list[str], ood: dict) -> str:
    if ood["is_ood"]:
        return (f"No defence object recognised. The image looks most like: {ood['closest_concept']}. "
                f"This system only recognises {', '.join(config.CLASSES.values())}.")
    label, conf = top3[0]["label"], top3[0]["confidence"]
    text = (f"Predicted {label} with {conf:.0%} confidence. The model associates this class with "
            f"{HINTS.get(label, 'its visual features')}; the heatmap shows the regions that most influenced "
            f"the decision. Second guess: {top3[1]['label']} ({top3[1]['confidence']:.0%}).")
    if reasons:
        text += " ⚠ Uncertain prediction: " + "; ".join(reasons) + ". Treat this result with caution."
    return text


# ---------------------------------------------------------------- Grad-CAM

def gradcam(clf: Classifier, img: Image.Image, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cam in [0,1] at model resolution, the RGB image exactly as the model saw it)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    bb = clf.backbone
    px = bb.preprocess([img])
    with torch.enable_grad():
        cam = GradCAM(model=clf, target_layers=[bb.cam_target_layer()], reshape_transform=bb.cam_reshape())
        grayscale = cam(input_tensor=px, targets=[ClassifierOutputTarget(class_idx)],
                        eigen_smooth=bb.cam_eigen_smooth)[0]
    seen = denormalise(bb, px[0])
    return grayscale, seen


def denormalise(bb, px: torch.Tensor) -> np.ndarray:
    mean, std = (np.array(v) for v in bb.mean_std())
    x = px.permute(1, 2, 0).numpy() * std + mean
    return np.clip(x, 0, 1)


def overlay(cam: np.ndarray, rgb: np.ndarray, alpha: float = 0.45, size: int = 448) -> Image.Image:
    heat = colormaps["jet"](cam)[..., :3]
    mix = (1 - alpha) * rgb + alpha * heat
    return Image.fromarray((mix * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)


def gradcam_png(clf: Classifier, img: Image.Image, class_idx: int) -> str | None:
    try:
        cam, seen = gradcam(clf, img, class_idx)
    except Exception as e:  # explanation is a bonus; never fail the prediction because of it
        print(f"[grad-cam] failed: {e}")
        return None
    return to_data_uri(overlay(cam, seen))


def to_data_uri(im: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=85)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()
