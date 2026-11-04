"""Tests for core.transformations."""

from __future__ import annotations

import numpy as np
import pytest

from core.transformations import (
    contrast_stretch,
    gamma_transform,
    identity_transform,
    linear_scaling,
    logarithmic_transform,
    negative_transform,
)


def test_identity_transform(sample_gray):
    out = identity_transform(sample_gray)
    assert out.dtype == np.float64
    assert np.allclose(out, sample_gray.astype(np.float64))


def test_linear_scaling_clips():
    A = np.array([[0, 100, 200]], dtype=np.uint8)
    out = linear_scaling(A, gain=2.0, offset=50.0)
    assert out[0, 0] == 50.0
    assert out[0, 1] == 250.0
    assert out[0, 2] == 255.0  # clipped


def test_negative_transform():
    A = np.array([[0, 128, 255]], dtype=np.uint8)
    out = negative_transform(A)
    assert np.array_equal(out, np.array([[255, 127, 0]], dtype=np.float64))


def test_logarithmic_transform():
    A = np.array([[0, 255]], dtype=np.uint8)
    out = logarithmic_transform(A, c=1.0)
    assert out[0, 0] == 0.0
    assert out[0, 1] == pytest.approx(np.log1p(255), rel=1e-6)


def test_gamma_transform_brighten_darken():
    A = np.array([[128]], dtype=np.uint8)
    brighter = gamma_transform(A, gamma=0.5)
    darker = gamma_transform(A, gamma=2.0)
    assert brighter[0, 0] > A[0, 0]
    assert darker[0, 0] < A[0, 0]


def test_gamma_invalid():
    with pytest.raises(ValueError):
        gamma_transform(np.zeros((2, 2)), gamma=0.0)


def test_contrast_stretch_full_range():
    A = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    out = contrast_stretch(A)
    assert out.min() == 0.0
    assert out.max() == 255.0


def test_contrast_stretch_constant():
    A = np.full((2, 2), 42, dtype=np.uint8)
    out = contrast_stretch(A)
    assert np.all(out == 0.0)