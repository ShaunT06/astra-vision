"""Helpers shared by the offline scripts (data prep, training, evaluation)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.validation import open_path  # noqa: E402

MANIFEST = ROOT / "dataset" / "manifest.csv"
EXCLUSIONS = ROOT / "dataset" / "exclusions.csv"
CACHE_DIR = config.DATA_DIR / "cache"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_manifest() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = read_csv(MANIFEST)
    y = np.array([config.CLASS_IDS.index(r["category"]) for r in rows])
    groups = np.array([r["group"] for r in rows])
    return rows, y, groups


def features(backbone_name: str, rows: list[dict], tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings for `rows` (+ horizontally-flipped copies for augmentation), cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{tag}_{backbone_name}.npz"
    names = np.array([r["file_name"] for r in rows])
    if path.exists():
        d = np.load(path)
        if len(d["names"]) == len(names) and (d["names"] == names).all():
            return d["X"], d["X_flip"]

    from app.models import get_backbone

    bb = get_backbone(backbone_name)
    images = [open_path(config.STARTER_DIR / r["file_name"]) for r in rows]
    print(f"  embedding {len(images)} images with {backbone_name} ...")
    X = bb.embed(images)
    X_flip = bb.embed([ImageOps.mirror(im) for im in images])
    np.savez(path, X=X, X_flip=X_flip, names=names)
    return X, X_flip
