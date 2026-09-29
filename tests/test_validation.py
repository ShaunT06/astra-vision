"""Bad-input handling. These tests need only Pillow (no models), so they run in seconds."""
import pytest
from PIL import Image

from app import config, validation
from app.validation import ImageError, load_image
from tests.conftest import encode


def expect(code, status, data):
    with pytest.raises(ImageError) as e:
        load_image(data)
    assert (e.value.code, e.value.status) == (code, status)


def test_valid_jpeg(jpeg_bytes):
    img = load_image(jpeg_bytes)
    assert img.mode == "RGB" and img.size == (320, 240)


@pytest.mark.parametrize("mode,fmt", [("RGBA", "PNG"), ("L", "PNG"), ("P", "PNG"), ("CMYK", "JPEG"),
                                      ("RGB", "WEBP"), ("RGB", "BMP"), ("P", "GIF"), ("I;16", "PNG")])
def test_colour_modes_and_formats_become_rgb(mode, fmt):
    img = load_image(encode(Image.new(mode, (64, 48)), fmt))
    assert img.mode == "RGB" and img.size == (64, 48)


def test_transparency_flattened_onto_white():
    img = load_image(encode(Image.new("RGBA", (40, 40), (0, 0, 0, 0)), "PNG"))
    assert img.getpixel((5, 5)) == (255, 255, 255)


def test_exif_rotation_applied():
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90° CW
    img = load_image(encode(Image.new("RGB", (100, 50)), "JPEG", exif=exif))
    assert img.size == (50, 100)


def test_empty_file():
    expect("empty_file", 422, b"")


def test_text_file_renamed_as_image():
    expect("unsupported_format", 415, b"this is definitely not a picture\n" * 20)


def test_truncated_jpeg(jpeg_bytes):
    expect("corrupt_image", 400, jpeg_bytes[: len(jpeg_bytes) // 2])


def test_garbage_after_valid_header(jpeg_bytes):
    expect("corrupt_image", 400, jpeg_bytes[:200] + b"\x00" * 500)


def test_disallowed_format():
    expect("unsupported_format", 415, encode(Image.new("RGB", (64, 64)), "ICO"))


def test_file_too_large(monkeypatch, jpeg_bytes):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    expect("file_too_large", 413, jpeg_bytes + b"\0" * 2000)


def test_decompression_bomb(monkeypatch):
    monkeypatch.setattr(config, "MAX_PIXELS", 1_000_000)
    monkeypatch.setattr(validation.Image, "MAX_IMAGE_PIXELS", 1_000_000)
    expect("image_too_large", 413, encode(Image.new("L", (2500, 2500)), "PNG"))


def test_too_small():
    expect("image_too_small", 422, encode(Image.new("RGB", (10, 10)), "PNG"))


def test_animated_gif_uses_first_frame():
    frames = [Image.new("P", (60, 60), i) for i in (10, 200)]
    buf = encode(frames[0], "GIF", save_all=True, append_images=frames[1:])
    assert load_image(buf).size == (60, 60)
