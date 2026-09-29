"""ASTRA VISION API. Frontend-agnostic JSON endpoints; interactive docs at /docs.

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import csv
import io
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .inference import classify
from .models import MODEL_INFO, available_models, get_classifier, load_metrics
from .validation import ImageError, load_image

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load weights at startup so the first user request isn't slow.
    for k in available_models():
        get_classifier(k)
    yield


app = FastAPI(
    title="ASTRA VISION API",
    description="Defence object recognition: classification, explanations, detection, history and metrics.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
config.REPORTS_DIR.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=config.REPORTS_DIR), name="reports")

# Grad-CAM registers hooks on shared model objects, so inference is serialised.
_infer_lock = threading.Lock()


@app.exception_handler(ImageError)
async def image_error_handler(_, exc: ImageError):
    return JSONResponse(status_code=exc.status, content={"error": exc.code, "message": exc.message})


def _check_model(model: str) -> str:
    if model not in MODEL_INFO:
        raise HTTPException(400, {"error": "unknown_model", "message": f"Choose one of {list(MODEL_INFO)}."})
    if model not in available_models():
        raise HTTPException(503, {"error": "model_not_trained", "message": f"{model} has no trained weights yet."})
    return model


def _predict_bytes(data: bytes, filename: str | None, model: str, explain: bool, save: bool) -> dict:
    img = load_image(data, filename)
    with _infer_lock:
        result = classify(img, model, explain=explain)
    result["filename"] = filename
    if save:
        result["history_id"] = db.save(result, img, filename)
    return result


@app.get("/api/")
def api_root():
    return {"message": "Sentinel Vision API online", "classification_endpoint": "/api/classify"}


def _region_box(clf, img, class_idx: int) -> dict | None:
    """Bounding box (percent of frame) around the Grad-CAM hot region. The CAM is computed on a
    square resize of the image, so its coordinates map directly to percentages of the frame."""
    import numpy as np

    from .inference import gradcam

    try:
        cam, _ = gradcam(clf, img, class_idx)
    except Exception as e:  # a box is a nicety; never fail classification because of it
        print(f"[classify] grad-cam box failed: {e}")
        return None
    ys, xs = np.where(cam >= 0.5 * cam.max())
    if len(xs) == 0:
        return None
    h, w = cam.shape
    x0, x1, y0, y1 = xs.min() / w, (xs.max() + 1) / w, ys.min() / h, (ys.max() + 1) / h
    return {"x": round(x0 * 100, 1), "y": round(y0 * 100, 1),
            "width": round((x1 - x0) * 100, 1), "height": round((y1 - y0) * 100, 1)}


@app.post("/api/classify")
async def classify_endpoint(
    file: UploadFile = File(...),
    model: str = Query(config.DEFAULT_MODEL),
    detect: bool = Query(False, description="Use the OWLv2 detector for one box per object (slower)"),
):
    """Sentinel Vision workbench endpoint: multipart `file` in, label + confidence + boxes out.
    Boxes are percentages of the frame. Runs the real models (no demo logic)."""
    _check_model(model)
    data, filename = await file.read(), file.filename
    img = load_image(data, filename)
    with _infer_lock:
        r = classify(img, model, explain=False)
        detections: list[dict] = []
        if r["defence_object_detected"]:
            if detect:
                from .detector import detect_objects

                W, H = img.size
                for d in detect_objects(img, model)["detections"]:
                    x0, y0, x1, y1 = d["box"]
                    detections.append({"label": d["label"].upper(), "confidence": d["confidence"],
                                       "x": round(x0 / W * 100, 1), "y": round(y0 / H * 100, 1),
                                       "width": round((x1 - x0) / W * 100, 1), "height": round((y1 - y0) / H * 100, 1)})
            if not detections:
                clf = get_classifier(model)
                idx = clf.labels.index(r["prediction"])
                box = _region_box(clf, img, idx)
                if box:
                    detections.append({"label": r["prediction"].upper(), "confidence": r["confidence"], **box})
    return {
        "id": f"SV-{uuid.uuid4().hex[:8].upper()}",
        "filename": filename,
        "label": (r["prediction"] or "NO DEFENCE OBJECT RECOGNISED").upper(),
        "confidence": r["confidence"],
        "detections": detections,
        "top3": r["top3"],
        "uncertain": r["uncertain"],
        "explanation": r["explanation"],
        "model": model,
        "latency_ms": r["latency_ms"],
        "mode": "LIVE INFERENCE",
        "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "models": available_models()}


@app.get("/api/models")
def models():
    metrics = load_metrics().get("models", {})
    return {
        "default": config.DEFAULT_MODEL,
        "classes": list(config.CLASSES.values()),
        "models": [
            {"key": k, "description": MODEL_INFO[k], "available": k in available_models(),
             "thresholds": config.load_thresholds(k),
             "accuracy": metrics.get(k, {}).get("accuracy_mean")}
            for k in MODEL_INFO
        ],
        "limits": {"max_upload_mb": config.MAX_UPLOAD_BYTES // 2**20, "max_batch_files": config.MAX_BATCH_FILES},
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(config.DEFAULT_MODEL),
    explain: bool = Query(True, description="Include a Grad-CAM heatmap"),
    save: bool = Query(True, description="Store the result in history"),
):
    _check_model(model)
    return _predict_bytes(await file.read(), file.filename, model, explain, save)


@app.post("/api/batch")
async def batch(
    files: list[UploadFile] = File(..., description="Several images, or one .zip of images"),
    model: str = Query(config.DEFAULT_MODEL),
    save: bool = Query(True),
):
    _check_model(model)
    items: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        if (f.filename or "").lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        if not info.is_dir() and not info.filename.startswith("__MACOSX"):
                            if info.file_size > config.MAX_UPLOAD_BYTES:
                                items.append((info.filename, b"\0" * 0))  # reported as too large below
                                continue
                            items.append((info.filename, z.read(info)))
            except zipfile.BadZipFile:
                raise ImageError("bad_zip", f"{f.filename} is not a valid zip archive.", 400)
        else:
            items.append((f.filename, data))
    if not items:
        raise ImageError("empty_batch", "No files were uploaded.", 422)
    if len(items) > config.MAX_BATCH_FILES:
        raise ImageError("batch_too_large", f"At most {config.MAX_BATCH_FILES} images per batch.", 413)

    results = []
    for name, data in items:
        try:
            r = _predict_bytes(data, name, model, explain=False, save=save)
            results.append({"filename": name, "ok": True, **{k: r[k] for k in
                            ("prediction", "confidence", "top3", "uncertain", "defence_object_detected",
                             "explanation", "history_id") if k in r}})
        except ImageError as e:  # one bad file never fails the whole batch
            results.append({"filename": name, "ok": False, "error": e.code, "message": e.message})

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["filename", "prediction", "confidence", "uncertain", "second", "third", "error"])
    for r in results:
        t = r.get("top3", [])
        w.writerow([r["filename"], r.get("prediction") or "", r.get("confidence", ""), r.get("uncertain", ""),
                    t[1]["label"] if len(t) > 1 else "", t[2]["label"] if len(t) > 2 else "", r.get("error", "")])
    ok = [r for r in results if r["ok"]]
    summary = {}
    for r in ok:
        key = r["prediction"] or "Not recognised"
        summary[key] = summary.get(key, 0) + 1
    return {"model": model, "count": len(results), "succeeded": len(ok), "summary": summary,
            "results": results, "csv": buf.getvalue()}


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    model: str = Query(config.DEFAULT_MODEL, description="Classifier applied to each detected box"),
    min_score: float = Query(0.15, ge=0.01, le=0.9),
):
    _check_model(model)
    from .detector import detect_objects

    img = load_image(await file.read(), file.filename)
    with _infer_lock:
        return detect_objects(img, model, min_score)


@app.get("/api/history")
def history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return {"items": db.list_recent(limit, offset)}


@app.get("/api/history/{item_id}")
def history_item(item_id: int):
    item = db.get(item_id)
    if not item:
        raise HTTPException(404, {"error": "not_found", "message": "No such history entry."})
    return item


@app.delete("/api/history/{item_id}")
def history_delete(item_id: int):
    if not db.delete(item_id):
        raise HTTPException(404, {"error": "not_found", "message": "No such history entry."})
    return {"deleted": item_id}


@app.delete("/api/history")
def history_clear():
    return {"deleted": db.clear()}


@app.get("/api/metrics")
def metrics():
    data = load_metrics()
    if not data:
        raise HTTPException(404, {"error": "no_metrics", "message": "Run `python -m scripts.evaluate` first."})
    data["confusion_images"] = {m: f"/reports/confusion_{m}.png" for m in data.get("models", {})}
    return data
