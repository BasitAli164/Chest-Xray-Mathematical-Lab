"""Tests for core.image_preprocessing.

Covers
------
- ensure_uint8 (no-op, clipping, rounding)
- ensure_grayscale (accept 2D, reject color, squeeze (H,W,1))
- resize_max_dimension (shrink, no-op, preserves aspect ratio)
- clip_to_range (bounds, dtype preservation, invalid bounds)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.image_preprocessing import (
    clip_to_range,
    ensure_grayscale,
    ensure_uint8,
    resize_max_dimension,
)


# ---------------------------------------------------------------------------
# ensure_uint8
# ---------------------------------------------------------------------------
def test_ensure_uint8_passthrough(sample_gray):
    out = ensure_uint8(sample_gray)
    assert out.dtype == np.uint8
    assert np.array_equal(out, sample_gray)


def test_ensure_uint8_clips_and_rounds():
    arr = np.array([[-5.4, 0.6], [127.5, 300.2]], dtype=np.float64)
    out = ensure_uint8(arr)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0        # -5.4 clipped to 0
    assert out[0, 1] == 1        # 0.6 rounded to 1
    assert out[1, 0] == 128      # 127.5 -> np.round -> 128
    assert out[1, 1] == 255      # 300.2 clipped to 255


# ---------------------------------------------------------------------------
# ensure_grayscale
# ---------------------------------------------------------------------------
def test_ensure_grayscale_accepts_2d(sample_gray):
    out = ensure_grayscale(sample_gray)
    assert out is sample_gray


def test_ensure_grayscale_squeezes_hw1():
    arr = np.zeros((10, 10, 1), dtype=np.uint8)
    out = ensure_grayscale(arr)
    assert out.shape == (10, 10)


def test_ensure_grayscale_rejects_rgb():
    with pytest.raises(ValueError, match="grayscale"):
        ensure_grayscale(np.zeros((4, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# resize_max_dimension
# ---------------------------------------------------------------------------
def test_resize_shrinks():
    img = np.zeros((200, 100), dtype=np.uint8)
    out = resize_max_dimension(img, 100)
    assert out.shape == (100, 50)
    assert out.dtype == img.dtype


def test_resize_no_op_when_within_limit():
    img = np.zeros((50, 40), dtype=np.uint8)
    out = resize_max_dimension(img, 100)
    assert out is img


def test_resize_invalid_max_dim():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_max_dimension(img, 0)


# ---------------------------------------------------------------------------
# clip_to_range
# ---------------------------------------------------------------------------
def test_clip_to_range_default():
    arr = np.array([-10, 0, 128, 255, 300], dtype=np.int32)
    out = clip_to_range(arr)
    assert out[0] == 0
    assert out[-1] == 255


def test_clip_to_range_preserves_uint8():
    arr = np.array([10, 200], dtype=np.uint8)
    out = clip_to_range(arr, 50, 150)
    assert out.dtype == np.uint8
    assert out[0] == 50
    assert out[1] == 150


def test_clip_to_range_invalid_bounds():
    with pytest.raises(ValueError):
        clip_to_range(np.zeros((2, 2)), lo=200, hi=100)