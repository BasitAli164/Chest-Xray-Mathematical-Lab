"""Tests for core.convolution."""

from __future__ import annotations

import numpy as np
import pytest

from core.convolution import (
    compare_implementations,
    convolve_manual,
    convolve_opencv,
    convolve_scipy,
    pad_image,
    validate_kernel,
)


def test_validate_kernel_ok():
    validate_kernel(np.ones((3, 3)))


def test_validate_kernel_rejects_even():
    with pytest.raises(ValueError):
        validate_kernel(np.ones((2, 2)))


def test_validate_kernel_rejects_1d():
    with pytest.raises(ValueError):
        validate_kernel(np.ones(3))


def test_validate_kernel_rejects_empty():
    with pytest.raises(ValueError):
        validate_kernel(np.zeros((0, 0)))


def test_pad_image_zero():
    A = np.ones((3, 3))
    P = pad_image(A, 1, 1, mode="zero")
    assert P.shape == (5, 5)
    assert P[0, 0] == 0
    assert P[1, 1] == 1


def test_pad_image_no_pad_returns_copy():
    A = np.ones((3, 3))
    P = pad_image(A, 0, 0)
    assert np.array_equal(P, A)
    assert P is not A


def test_pad_image_reflect():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    P = pad_image(A, 1, 1, mode="reflect")
    assert P.shape == (4, 4)
    assert P[1, 1] == 1          # center = original A[0, 0]
    # np.pad 'reflect' mirrors without repeating the edge:
    assert P[0, 0] == 4.0        # = A[1, 1]
    assert P[0, 1] == 3.0        # = A[1, 0]
    assert P[1, 0] == 2.0        # = A[0, 1]


def test_pad_image_invalid_mode():
    with pytest.raises(ValueError):
        pad_image(np.zeros((3, 3)), 1, 1, mode="not_a_mode")


def test_manual_convolution_identity_kernel():
    A = np.arange(25, dtype=np.float64).reshape(5, 5)
    K = np.zeros((3, 3))
    K[1, 1] = 1.0
    out = convolve_manual(A, K, padding="reflect")
    assert np.allclose(out, A)


def test_manual_convolution_matches_scipy():
    """Manual convolution implements cross-correlation (kernel NOT flipped),
    matching image-processing convention. SciPy's convolve2d flips the
    kernel. Compare against scipy.signal.correlate2d on the interior only
    (boundary handling differs between libraries)."""
    from scipy.signal import correlate2d

    rng = np.random.default_rng(0)
    A = rng.integers(0, 255, size=(10, 10)).astype(np.float64)
    K = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    m = convolve_manual(A, K, padding="reflect")
    c = correlate2d(A, K, mode="same", boundary="fill")
    interior_m = m[1:-1, 1:-1]
    interior_c = c[1:-1, 1:-1]
    assert np.allclose(interior_m, interior_c)

    rng = np.random.default_rng(0)
    A = rng.integers(0, 255, size=(10, 10)).astype(np.float64)
    K = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    m = convolve_manual(A, K, padding="reflect")
    c = correlate2d(A, K, mode="same", boundary="symm")
    # Boundary handling differs slightly between implementations.
    # Interior should match exactly.
    interior_m = m[1:-1, 1:-1]
    interior_c = c[1:-1, 1:-1]
    assert np.allclose(interior_m, interior_c)


def test_manual_convolution_mean_kernel_constant_image():
    A = np.full((5, 5), 100, dtype=np.float64)
    K = np.full((3, 3), 1.0 / 9.0)
    out = convolve_manual(A, K, padding="reflect")
    assert np.allclose(out, 100.0)


def test_compare_implementations_shape():
    A = np.arange(64, dtype=np.float64).reshape(8, 8)
    K = np.ones((3, 3)) / 9.0
    report = compare_implementations(A, K, padding="reflect")
    assert "manual_vs_scipy_max_abs_diff" in report
    assert report["manual_shape"] == [8, 8]


def test_convolve_rejects_1d():
    with pytest.raises(ValueError):
        convolve_manual(np.zeros(5), np.ones((3, 3)))