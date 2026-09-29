"""Central configuration. Everything tunable lives here; values can be overridden by env vars."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional .env support; real environment variables still win
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = ROOT / "data"
STARTER_DIR = DATA_DIR / "starter" / "ASTRA-Challenge-Starter" / "challenge-02-vision"
ARTIFACTS_DIR = ROOT / "models"             # trained heads, calibration, thresholds
REPORTS_DIR = ROOT / "reports"              # metrics.json, confusion matrices
STORAGE_DIR = Path(os.getenv("ASTRA_STORAGE_DIR", ROOT / "storage"))  # history db + thumbnails

# Class id (folder name in the dataset) -> human-readable label, in a fixed order.
CLASSES: dict[str, str] = {
    "aircraft": "Military Aircraft",
    "helicopter": "Military Helicopter",
    "drone": "Drone / UAV",
    "military-vehicle": "Military Land Vehicle",
    "naval": "Naval Vessel",
}
CLASS_IDS = list(CLASSES)

# Text prompts for zero-shot classification (averaged per class = prompt ensembling).
CLASS_PROMPTS: dict[str, list[str]] = {
    "aircraft": [
        "a photo of a military fighter jet",
        "a photo of a military aircraft",
        "a photo of a warplane in flight",
        "a photo of a combat aircraft on a runway",
    ],
    "helicopter": [
        "a photo of a military helicopter",
        "a photo of an attack helicopter",
        "a photo of a transport helicopter with rotor blades",
    ],
    "drone": [
        "a photo of a drone",
        "a photo of a quadcopter drone",
        "a photo of an unmanned aerial vehicle",
        "a photo of a military UAV",
    ],
    "military-vehicle": [
        "a photo of a military tank",
        "a photo of an armored military vehicle",
        "a photo of an armored personnel carrier",
        "a photo of a military truck",
    ],
    "naval": [
        "a photo of a warship",
        "a photo of a naval vessel at sea",
        "a photo of a navy destroyer",
        "a photo of a submarine",
    ],
}

# Prompts describing things that are NOT one of our defence classes. Used by the
# out-of-distribution gate: if one of these wins over every defence prompt, we say so.
NON_DEFENCE_PROMPTS: dict[str, list[str]] = {
    "animal": ["a photo of an animal", "a photo of a cat", "a photo of a dog", "a photo of a bird"],
    "person": ["a photo of a person", "a selfie", "a portrait of a person"],
    "civilian airliner": ["a photo of a commercial passenger airliner"],
    "civilian car": ["a photo of a civilian car", "a photo of a car on a road"],
    "civilian ship": ["a photo of a cruise ship", "a photo of a cargo container ship", "a photo of a sailboat"],
    "food": ["a photo of food"],
    "landscape": ["a photo of a landscape", "a photo of a city street", "an aerial photo of a town"],
    "document or diagram": ["a screenshot of text", "a map", "a technical diagram"],
    "indoor scene": ["a photo of a room interior"],
}

# Model identifiers (all open-weight, free).
SIGLIP_MODEL_ID = os.getenv("ASTRA_SIGLIP_MODEL", "google/siglip2-base-patch16-224")
EFFNET_MODEL_ID = os.getenv("ASTRA_EFFNET_MODEL", "efficientnet_b0.ra_in1k")
OWL_MODEL_ID = os.getenv("ASTRA_OWL_MODEL", "google/owlv2-base-patch16-ensemble")

DEFAULT_MODEL = os.getenv("ASTRA_DEFAULT_MODEL", "siglip-probe")

# Input limits (graceful bad-input handling).
MAX_UPLOAD_BYTES = int(os.getenv("ASTRA_MAX_UPLOAD_MB", "15")) * 1024 * 1024
MAX_PIXELS = int(os.getenv("ASTRA_MAX_PIXELS", str(50_000_000)))   # decompression-bomb guard
MIN_SIDE = 32                                                        # too small to classify
MAX_BATCH_FILES = int(os.getenv("ASTRA_MAX_BATCH", "25"))
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF", "MPO"}

# Uncertainty defaults. Overridden by models/thresholds.json produced from validation data.
DEFAULT_THRESHOLDS = {"min_confidence": 0.55, "min_margin": 0.15, "temperature": 1.0}


def load_thresholds(model_name: str) -> dict:
    path = ARTIFACTS_DIR / "thresholds.json"
    if path.exists():
        data = json.loads(path.read_text())
        if model_name in data:
            return {**DEFAULT_THRESHOLDS, **data[model_name]}
    return dict(DEFAULT_THRESHOLDS)
