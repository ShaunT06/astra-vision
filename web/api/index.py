"""Sentinel Vision API on Vercel: SigLIP 2 (int8 ONNX) + trained probe head, no PyTorch.

GET  /api/          -> service banner
POST /api/classify  -> multipart `file`; label, confidence, detections (percent boxes)
Mirrors the FastAPI backend in app/main.py so the frontend is identical for both.
"""
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
MAX_BYTES = 4 * 1024 * 1024        # Vercel request-body limit is ~4.5 MB
MAX_PIXELS = 50_000_000
HINTS = {
    "Military Aircraft": "fixed wings, jet or propeller engines and a streamlined fuselage",
    "Military Helicopter": "a main rotor, tail boom and cabin-shaped fuselage",
    "Drone / UAV": "a small unmanned airframe, often with multiple rotors or no cockpit",
    "Military Land Vehicle": "tracks or large wheels, armour plating and often a turret",
    "Naval Vessel": "a hull on water, superstructure and deck equipment",
}

_state = {}


def _load():
    if _state:
        return _state
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    _state["sess"] = ort.InferenceSession(os.path.join(MODEL_DIR, "siglip_vision.onnx"), so,
                                          providers=["CPUExecutionProvider"])
    h = np.load(os.path.join(MODEL_DIR, "heads.npz"))
    _state["h"] = {k: h[k] for k in h.files}
    _state["th"] = json.loads(str(h["thresholds"]))
    return _state


class BadImage(Exception):
    def __init__(self, code, message, status=400):
        self.code, self.message, self.status = code, message, status


def _load_image(data: bytes) -> Image.Image:
    if not data:
        raise BadImage("empty_file", "The uploaded file is empty.", 422)
    if len(data) > MAX_BYTES:
        raise BadImage("file_too_large", "Image exceeds the 4 MB upload limit.", 413)
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise BadImage("corrupt_image", "That file could not be read as an image.", 415)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < 32:
        raise BadImage("image_too_small", "Image is too small to classify.", 422)
    return img


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _embed(st, img):
    x = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    pooled = st["sess"].run(["pooled"], {"pixel_values": x})[0][0]
    return pooled / np.linalg.norm(pooled)


def _classify(img: Image.Image) -> dict:
    st = _load()
    h = st["h"]
    t0 = time.perf_counter()
    feat = _embed(st, img)

    th = st["th"]
    probs = _softmax((h["probe_W"] @ feat + h["probe_b"]) / th.get("temperature", 1.0))
    labels = [str(s) for s in h["probe_labels"]]
    order = np.argsort(probs)[::-1]
    top3 = [{"label": labels[i], "confidence": round(float(probs[i]), 4)} for i in order[:3]]
    best, second = float(probs[order[0]]), float(probs[order[1]])
    reasons = []
    if best < th["min_confidence"]:
        reasons.append(f"top confidence {best:.0%} is below the {th['min_confidence']:.0%} threshold")
    if best - second < th["min_margin"]:
        reasons.append(f"the top two classes are only {best - second:.0%} apart")

    p_ood = _softmax(h["ood_W"] @ feat + h["ood_b"])
    ood_labels = [str(s) for s in h["ood_labels"]]
    closest = ood_labels[int(np.argmax(p_ood))]
    is_ood = closest.startswith("other:")

    detections = []
    if not is_ood:
        # The probe classifies the whole frame, so the honest region is the whole frame. Per-object
        # boxes need the OWLv2 detector (backend `?detect=true`), which is too large for a Vercel function.
        detections.append({"label": top3[0]["label"].upper(), "confidence": top3[0]["confidence"],
                           "x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0, "scope": "frame"})

    if is_ood:
        expl = (f"No defence object recognised. The image looks most like: {closest.split(':', 1)[1]}. "
                f"This system only recognises {', '.join(HINTS)}.")
    else:
        expl = (f"Predicted {top3[0]['label']} with {top3[0]['confidence']:.0%} confidence. The model associates "
                f"this class with {HINTS.get(top3[0]['label'], 'its visual features')}. "
                f"Second guess: {top3[1]['label']} ({top3[1]['confidence']:.0%}).")
        if reasons:
            expl += " Uncertain prediction: " + "; ".join(reasons) + ". Treat this result with caution."
    return {
        "id": f"SV-{uuid.uuid4().hex[:8].upper()}",
        "label": "NO DEFENCE OBJECT RECOGNISED" if is_ood else top3[0]["label"].upper(),
        "confidence": top3[0]["confidence"],
        "detections": detections,
        "top3": top3,
        "uncertain": bool(reasons) or is_ood,
        "explanation": expl,
        "model": "siglip-probe (onnx-int8)",
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "mode": "LIVE INFERENCE",
        "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _multipart_file(headers, body: bytes):
    ctype = headers.get("Content-Type", "")
    if not ctype.startswith("multipart/form-data"):
        raise BadImage("bad_request", "Send the image as multipart/form-data with field 'file'.", 415)
    msg = BytesParser(policy=policy.HTTP).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    for part in msg.iter_parts():
        if part.get_param("name", header="content-disposition") == "file":
            return part.get_filename(), part.get_payload(decode=True) or b""
    raise BadImage("no_file", "Missing multipart field 'file'.", 422)


class handler(BaseHTTPRequestHandler):
    def _send(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        self._send(200, {"message": "Sentinel Vision API online", "classification_endpoint": "/api/classify"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BYTES + 65536:
                raise BadImage("file_too_large", "Image exceeds the 4 MB upload limit.", 413)
            filename, data = _multipart_file(self.headers, self.rfile.read(n))
            result = _classify(_load_image(data))
            result["filename"] = filename
            self._send(200, result)
        except BadImage as e:
            self._send(e.status, {"error": e.code, "message": e.message})
        except Exception as e:  # never leak a stack trace to the client
            print(f"[classify] {type(e).__name__}: {e}")
            self._send(500, {"error": "inference_failed", "message": "The classifier failed on this frame."})
