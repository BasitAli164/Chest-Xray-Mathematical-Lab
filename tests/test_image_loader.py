"""Tests for core.image_loader.

Covers
------
- extension validation (ok + bad cases)
- size validation (empty + oversize)
- decoding failure
- dimension limits (too small)
- grayscale conversion (rgb, gray, rgba, 16-bit)
- metadata correctness
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from core.image_loader import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    build_metadata,
    load_image,
    to_grayscale,
    validate_extension,
    validate_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _encode(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    """Encode a numpy array into the specified image format."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# validate_extension
# ---------------------------------------------------------------------------
def test_validate_extension_ok():
    assert validate_extension("xray.PNG") == "png"
    assert validate_extension("scan.jpeg") == "jpeg"
    assert validate_extension("a.b.jpg") == "jpg"


def test_validate_extension_no_extension():
    with pytest.raises(ValueError, match="no extension"):
        validate_extension("noext")


def test_validate_extension_unsupported():
    with pytest.raises(ValueError, match="Unsupported extension"):
        validate_extension("x.tif")


def test_allowed_extensions_constant():
    assert set(ALLOWED_EXTENSIONS) == {"png", "jpg", "jpeg"}


# ---------------------------------------------------------------------------
# validate_size
# ---------------------------------------------------------------------------
def test_validate_size_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_size(b"")


def test_validate_size_oversize():
    oversize = b"\x00" * (MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="too large"):
        validate_size(oversize)


# ---------------------------------------------------------------------------
# load_image — error paths
# ---------------------------------------------------------------------------
def test_load_image_rejects_small():
    tiny = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        load_image(_encode(tiny), "t.png")


def test_load_image_rejects_bad_bytes():
    with pytest.raises(ValueError, match="could not be decoded"):
        load_image(b"not an image at all", "bad.png")


def test_load_image_rejects_oversize():
    with pytest.raises(ValueError, match="too large"):
        load_image(b"\x00" * (MAX_FILE_SIZE_MB * 1024 * 1024 + 1), "big.png")


# ---------------------------------------------------------------------------
# load_image — success paths
# ---------------------------------------------------------------------------
def test_load_image_valid_gray(sample_gray):
    arr = load_image(_encode(sample_gray), "x.png")
    assert arr.shape == sample_gray.shape
    assert arr.dtype == np.uint8
    assert np.array_equal(arr, sample_gray)


def test_load_image_valid_rgb(sample_rgb):
    arr = load_image(_encode(sample_rgb), "x.png")
    assert arr.shape == sample_rgb.shape
    assert arr.dtype == np.uint8


def test_load_image_jpeg(sample_rgb):
    arr = load_image(_encode(sample_rgb, "JPEG"), "x.jpg")
    assert arr.shape == sample_rgb.shape


# ---------------------------------------------------------------------------
# to_grayscale
# ---------------------------------------------------------------------------
def test_to_grayscale_rgb(sample_rgb):
    g = to_grayscale(sample_rgb)
    assert g.shape == sample_rgb.shape[:2]
    assert g.dtype == np.uint8

    # Pixel at (20, 20): RGB = (255, 0, 0) because green is only top half
    # and blue is only left half.  0.299*255 = 76.245 -> round = 76
    assert g[20, 20] == pytest.approx(76, abs=1)


def test_to_grayscale_already_gray(sample_gray):
    g = to_grayscale(sample_gray)
    assert np.array_equal(g, sample_gray)
    assert g is not sample_gray  # copy, not reference


def test_to_grayscale_rgba(sample_rgba):
    g = to_grayscale(sample_rgba)
    assert g.shape == sample_rgba.shape[:2]
    assert g.dtype == np.uint8


def test_to_grayscale_16bit():
    # 16-bit grayscale (dtype uint16)
    img = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)
    g = to_grayscale(img)
    assert g.dtype == np.uint8
    assert g.min() == 0
    assert g.max() == 255


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------
def test_build_metadata(sample_gray):
    meta = build_metadata(sample_gray, sample_gray)
    assert meta.width == 64
    assert meta.height == 64
    assert meta.array_shape == (64, 64)
    assert meta.array_dtype == "uint8"
    assert meta.intensity_min == int(sample_gray.min())
    assert meta.intensity_max == int(sample_gray.max())
    assert meta.intensity_mean == pytest.approx(float(sample_gray.mean()))
    assert meta.intensity_median == pytest.approx(
        float(np.median(sample_gray))
    )
    assert meta.intensity_std == pytest.approx(float(sample_gray.std()))