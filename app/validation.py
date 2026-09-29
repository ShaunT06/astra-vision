"""Input validation + preprocessing shared by the API and the offline scripts.

Every uploaded file goes through `load_image`, which either returns a clean RGB
PIL image or raises `ImageError` with a machine-readable code and HTTP status.
"""
from __future__ import annotations

import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

# Pillow's own bomb guard warns at MAX_IMAGE_PIXELS and errors at 2x; we enforce our own
# limit explicitly below, so make Pillow error out instead of silently allocating.
Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS


class ImageError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def load_image(data: bytes, filename: str | None = None) -> Image.Image:
    if not data:
        raise ImageError("empty_file", "The uploaded file is empty.", 422)
    if len(data) > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ImageError("file_too_large", f"File is larger than the {mb} MB limit.", 413)

    # Pass 1: verify() detects truncated/corrupt files without decoding everything.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            fmt = probe.format
            probe.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError("image_too_large", "Image has too many pixels to process safely.", 413)
    except UnidentifiedImageError:
        if _has_image_signature(data):   # starts like an image but can't be parsed -> damaged
            raise ImageError("corrupt_image", "The image file is corrupt or truncated.", 400)
        raise ImageError("unsupported_format", "File is not a recognised image format.", 415)
    except Exception:
        raise ImageError("corrupt_image", "The image file is corrupt or truncated.", 400)

    if fmt not in config.ALLOWED_FORMATS:
        raise ImageError("unsupported_format", f"Image format {fmt} is not supported.", 415)

    # Pass 2: verify() leaves the image unusable, so reopen and fully decode.
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > config.MAX_PIXELS:
            raise ImageError("image_too_large", "Image has too many pixels to process safely.", 413)
        img.load()
    except ImageError:
        raise
    except Exception:
        raise ImageError("corrupt_image", "The image could not be decoded (corrupt or truncated).", 400)

    return normalise(img)


SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM", b"II*\x00", b"MM\x00*")


def _has_image_signature(data: bytes) -> bool:
    return data.startswith(SIGNATURES) or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")


def normalise(img: Image.Image) -> Image.Image:
    """Fix orientation and colour mode so every model sees a plain RGB image."""
    img = ImageOps.exif_transpose(img)            # phone photos store rotation in EXIF
    if getattr(img, "is_animated", False):
        img.seek(0)                                # animated GIF/WebP: use first frame
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))   # flatten transparency onto white
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")                   # grayscale, CMYK, 16-bit, ...
    if min(img.size) < config.MIN_SIDE:
        raise ImageError("image_too_small", f"Image is smaller than {config.MIN_SIDE}px on one side.", 422)
    return img


def open_path(path) -> Image.Image:
    with open(path, "rb") as f:
        return load_image(f.read(), str(path))
