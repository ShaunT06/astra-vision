"""End-to-end API tests. These load the real models, so they are slower (~1 min).
Skipped automatically if torch can't be imported or the heads aren't trained yet."""
import io
import zipfile

import pytest

torch = pytest.importorskip("torch")

from app import config  # noqa: E402

if not (config.ARTIFACTS_DIR / "siglip-probe.npz").exists():
    pytest.skip("run `python -m scripts.train` first", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402
from tests.conftest import encode  # noqa: E402

client = TestClient(app)
SAMPLE = next((config.STARTER_DIR / "images" / "helicopter").glob("*.jpg"))


def post(path, data, name="x.jpg", **params):
    return client.post(path, files={"file": (name, data)}, params=params)


def test_health_and_models():
    assert client.get("/api/health").json()["status"] == "ok"
    body = client.get("/api/models").json()
    assert len(body["classes"]) == 5
    assert {m["key"] for m in body["models"]} >= {"siglip-probe", "siglip-zeroshot", "effnet-probe"}


@pytest.mark.parametrize("model", ["siglip-probe", "siglip-zeroshot", "effnet-probe"])
def test_predict_real_image(model):
    r = post("/api/predict", SAMPLE.read_bytes(), SAMPLE.name, model=model, save=False)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["top3"]) == 3
    assert abs(sum(t["confidence"] for t in body["top3"])) <= 1.0001
    assert body["top3"][0]["confidence"] >= body["top3"][1]["confidence"] >= body["top3"][2]["confidence"]
    assert body["explanation"]
    assert body["heatmap"] is None or body["heatmap"].startswith("data:image/")


def test_probe_gets_training_image_right():
    body = post("/api/predict", SAMPLE.read_bytes(), model="siglip-probe", save=False).json()
    assert body["prediction"] == "Military Helicopter"


def test_blank_image_is_not_confidently_classified():
    body = post("/api/predict", encode(Image.new("RGB", (300, 300), (128, 128, 128)), "PNG"),
                "grey.png", save=False).json()
    assert body["uncertain"] is True


@pytest.mark.parametrize("data,status,code", [
    (b"", 422, "empty_file"),
    (b"not an image at all" * 10, 415, "unsupported_format"),
    (b"\xff\xd8\xff" + b"\x00" * 300, 400, "corrupt_image"),
])
def test_predict_bad_inputs(data, status, code):
    r = post("/api/predict", data)
    assert r.status_code == status and r.json()["error"] == code


def test_unknown_model():
    r = post("/api/predict", SAMPLE.read_bytes(), model="nope")
    assert r.status_code == 400


def test_batch_with_zip_and_bad_file():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.jpg", SAMPLE.read_bytes())
        z.writestr("broken.jpg", b"garbage")
    files = [("files", ("set.zip", buf.getvalue())), ("files", ("b.jpg", SAMPLE.read_bytes()))]
    body = client.post("/api/batch", files=files, params={"save": False}).json()
    assert body["count"] == 3 and body["succeeded"] == 2
    assert "filename,prediction" in body["csv"]


def test_history_roundtrip():
    item = post("/api/predict", SAMPLE.read_bytes(), SAMPLE.name, explain=False).json()
    hid = item["history_id"]
    assert client.get(f"/api/history/{hid}").json()["prediction"] == item["prediction"]
    assert client.delete(f"/api/history/{hid}").status_code == 200
    assert client.get(f"/api/history/{hid}").status_code == 404


def test_api_root():
    assert client.get("/api/").json() == {"message": "Sentinel Vision API online",
                                          "classification_endpoint": "/api/classify"}


def test_classify_shape():
    body = post("/api/classify", SAMPLE.read_bytes(), SAMPLE.name).json()
    assert body["label"] == "MILITARY HELICOPTER" and body["mode"] == "LIVE INFERENCE"
    assert body["id"].startswith("SV-") and body["processed_at"].endswith("Z")
    d = body["detections"][0]
    assert all(0 <= d[k] <= 100 for k in ("x", "y", "width", "height"))


def test_classify_rejects_bad_file():
    assert post("/api/classify", b"not an image").status_code in (400, 415)
