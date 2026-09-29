"""Compare the serverless ONNX classifier (web/api/index.py) with the PyTorch one on starter images.

Run:  python -m scripts.check_onnx_parity
"""
import importlib.util
import random

from PIL import Image

from app import config
from app.inference import classify

spec = importlib.util.spec_from_file_location("vercel_api", config.ROOT / "web" / "api" / "index.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

random.seed(1)
imgs = []
for d in sorted((config.STARTER_DIR / "images").iterdir()):
    files = sorted(d.glob("*.*"))
    imgs += random.sample(files, min(6, len(files)))

agree = 0
for p in imgs:
    im = Image.open(p).convert("RGB")
    t = classify(im, "siglip-probe", explain=False)
    o = api._classify(im)
    same = (t["prediction"] or "NO DEFENCE OBJECT RECOGNISED").upper() == o["label"]
    agree += same
    print(f"{'OK ' if same else 'DIFF'} {p.parent.name:17s} torch={t['prediction']} {t['confidence']:.3f} | "
          f"onnx={o['label']} {o['confidence']:.3f} box={o['detections'][:1] and (o['detections'][0]['x'], o['detections'][0]['y'], o['detections'][0]['width'], o['detections'][0]['height'])} {o['latency_ms']}ms")
print(f"label agreement: {agree}/{len(imgs)}")
