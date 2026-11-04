"""Tests for core.enhancement."""

from __future__ import annotations

import numpy as np
import pytest

from core.enhancement import (
    adjust_brightness,
    adjust_contrast,
    apply_clahe,
    compute_histogram,
    histogram_equalization_manual,
    histogram_equalization_opencv,
    min_max_normalize,
    sharpen,
)


def test_adjust_brightness_positive(sample_gray):
    out = adjust_brightness(sample_gray, offset=30)
    assert out.dtype == np.uint8
    assert out.shape == sample_gray.shape
    assert int(out[0, 0]) == min(255, int(sample_gray[0, 0]) + 30)


def test_adjust_brightness_clips():
    A = np.array([[250, 10]], dtype=np.uint8)
    out = adjust_brightness(A, offset=20)
    assert out[0, 0] == 255
    assert out[0, 1] == 30


def test_adjust_brightness_negative():
    A = np.array([[5, 100]], dtype=np.uint8)
    out = adjust_brightness(A, offset=-20)
    assert out[0, 0] == 0
    assert out[0, 1] == 80


def test_adjust_contrast_gain_gt_one(sample_gray):
    out = adjust_contrast(sample_gray, gain=2.0)
    assert out.dtype == np.uint8
    assert out.std() >= sample_gray.std()


def test_adjust_contrast_gain_identity():
    A = np.array([[100, 200]], dtype=np.uint8)
    out = adjust_contrast(A, gain=1.0)
    # (1.0*(100-128)+128) = 100, exact
    assert out[0, 0] == 100
    assert out[0, 1] == 200


def test_min_max_normalize_stretches():
    A = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    out = min_max_normalize(A)
    assert out.min() == 0
    assert out.max() == 255


def test_min_max_normalize_constant():
    A = np.full((3, 3), 42, dtype=np.uint8)
    out = min_max_normalize(A)
    assert np.all(out == 0)


def test_hist_eq_manual_matches_opencv(sample_gray):
    manual = histogram_equalization_manual(sample_gray)
    cv = histogram_equalization_opencv(sample_gray)
    assert manual.shape == cv.shape
    # Allow ±2 rounding difference
    assert int(np.max(np.abs(manual.astype(int) - cv.astype(int)))) <= 2


def test_hist_eq_manual_constant():
    A = np.full((5, 5), 128, dtype=np.uint8)
    out = histogram_equalization_manual(A)
    assert np.all(out == 0)


def test_clahe_returns_same_shape(sample_gray):
    out = apply_clahe(sample_gray, clip_limit=2.0, tile_grid_size=(8, 8))
    assert out.shape == sample_gray.shape
    assert out.dtype == np.uint8


def test_clahe_invalid_clip():
    with pytest.raises(ValueError):
        apply_clahe(np.zeros((10, 10), dtype=np.uint8), clip_limit=0.0)


def test_sharpen_preserves_shape(sample_gray):
    out = sharpen(sample_gray, strength=1.0)
    assert out.shape == sample_gray.shape
    assert out.dtype == np.uint8


def test_sharpen_zero_strength_identity(sample_gray):
    out = sharpen(sample_gray, strength=0.0)
    assert np.allclose(out, sample_gray)


def test_sharpen_negative_strength_raises(sample_gray):
    with pytest.raises(ValueError):
        sharpen(sample_gray, strength=-1.0)


def test_compute_histogram(sample_gray):
    h = compute_histogram(sample_gray, bins=256)
    assert len(h["counts"]) == 256
    assert sum(h["counts"]) == sample_gray.size