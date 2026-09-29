"""Prediction history (gallery) in SQLite. Thumbnails are stored inline as small JPEG data URIs."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from PIL import Image

from . import config
from .inference import to_data_uri

_lock = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT,
    model TEXT NOT NULL,
    prediction TEXT,
    confidence REAL,
    uncertain INTEGER,
    defence_object_detected INTEGER,
    top3 TEXT,
    explanation TEXT,
    thumbnail TEXT
)"""


def _conn() -> sqlite3.Connection:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.STORAGE_DIR / "history.db")
    c.row_factory = sqlite3.Row
    c.execute(SCHEMA)
    return c


def save(result: dict, img: Image.Image, filename: str | None) -> int:
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (created_at, filename, model, prediction, confidence, uncertain,"
            " defence_object_detected, top3, explanation, thumbnail) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), filename, result["model"],
             result["prediction"], result["confidence"], int(result["uncertain"]),
             int(result["defence_object_detected"]), json.dumps(result["top3"]), result["explanation"],
             to_data_uri(thumb)),
        )
        return cur.lastrowid


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["top3"] = json.loads(d["top3"] or "[]")
    d["uncertain"] = bool(d["uncertain"])
    d["defence_object_detected"] = bool(d["defence_object_detected"])
    return d


def list_recent(limit: int = 50, offset: int = 0) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        return [_row(r) for r in rows]


def get(item_id: int) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM predictions WHERE id = ?", (item_id,)).fetchone()
        return _row(r) if r else None


def delete(item_id: int) -> bool:
    with _lock, _conn() as c:
        return c.execute("DELETE FROM predictions WHERE id = ?", (item_id,)).rowcount > 0


def clear() -> int:
    with _lock, _conn() as c:
        return c.execute("DELETE FROM predictions").rowcount
