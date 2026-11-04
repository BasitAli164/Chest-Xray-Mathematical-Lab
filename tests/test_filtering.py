"""Tests for core.filtering."""

from __future__ import annotations

import numpy as np
import pytest

from core.filtering import (
    gaussian_filter,
    gaussian_kernel,
    mean_filter,
    mean_kernel,
    median_filter,
)


def test_mean_kernel_sum():
    K = mean_kernel(3)
    assert K.shape == (3, 3)
    assert K.sum() == pytest.approx(1.0)


def test_mean_kernel_invalid_size():
    with pytest.raises(ValueError):
        mean_kernel(2)


def test_gaussian_kernel_normalized():
    K = gaussian_kernel(5, sigma=1.0)
    assert K.shape == (5, 5)
    assert K.sum() == pytest.approx(1.0)


def test_gaussian_kernel_peak_at_center():
    K = gaussian_kernel(5, sigma=1.0)
    assert K[2, 2] == K.max()


def test_gaussian_kernel_invalid():
    with pytest.raises(ValueError):
        gaussian_kernel(4, sigma=1.0)
    with pytest.raises(ValueError):
        gaussian_kernel(3, sigma=0.0)


def test_mean_filter_preserves_shape(sample_gray):
    out = mean_filter(sample_gray, kernel_size=3)
    assert out.shape == sample_gray.shape
    assert out.dtype == np.uint8


def test_mean_filter_blurs(sample_gray):
    out = mean_filter(sample_gray, kernel_size=5)
    # Blurring reduces high-frequency energy -> lower std
    assert out.std() <= sample_gray.std()


def test_gaussian_filter_shape(sample_gray):
    out = gaussian_filter(sample_gray, kernel_size=5, sigma=1.0)
    assert out.shape == sample_gray.shape


def test_gaussian_filter_invalid_sigma(sample_gray):
    with pytest.raises(ValueError):
        gaussian_filter(sample_gray, kernel_size=3, sigma=0.0)


def test_median_filter_shape(sample_gray):
    out = median_filter(sample_gray, kernel_size=3)
    assert out.shape == sample_gray.shape


def test_median_filter_removes_salt_pepper():
    A = np.full((20, 20), 100, dtype=np.uint8)
    A[5, 5] = 0
    A[10, 10] = 255
    out = median_filter(A, kernel_size=3)
    assert out[5, 5] == 100
    assert out[10, 10] == 100