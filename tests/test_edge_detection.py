"""Tests for core.edge_detection."""

from __future__ import annotations

import numpy as np
import pytest

from core.edge_detection import (
    LAPLACIAN_4,
    LAPLACIAN_8,
    SOBEL_X,
    SOBEL_Y,
    canny,
    edge_density,
    gradient_magnitude,
    laplacian,
    laplacian_uint8,
    sobel_uint8,
    sobel_x,
    sobel_y,
)


def test_kernel_shapes():
    assert SOBEL_X.shape == (3, 3)
    assert SOBEL_Y.shape == (3, 3)
    assert LAPLACIAN_4.shape == (3, 3)
    assert LAPLACIAN_8.shape == (3, 3)


def test_sobel_x_signed_is_float():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    gx = sobel_x(A)
    assert gx.dtype == np.float64
    assert gx.shape == A.shape


def test_sobel_y_signed_is_float():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    gy = sobel_y(A)
    assert gy.dtype == np.float64


def test_sobel_x_vertical_edge():
    # A vertical edge at column 5 -> strong gx response near boundary
    A = np.zeros((10, 10), dtype=np.uint8)
    A[:, 5:] = 255
    gx = sobel_x(A)
    # Column ~4-5 should show large response
    assert abs(gx[5, 5]) > 100 or abs(gx[5, 4]) > 100


def test_sobel_uint8_returns_uint8(sample_gray):
    out = sobel_uint8(sample_gray, axis="x")
    assert out.dtype == np.uint8
    assert out.shape == sample_gray.shape


def test_gradient_magnitude_pythagorean():
    gx = np.array([[3.0, 0.0]])
    gy = np.array([[4.0, 5.0]])
    mag = gradient_magnitude(gx, gy)
    assert mag[0, 0] == pytest.approx(5.0)
    assert mag[0, 1] == pytest.approx(5.0)


def test_gradient_magnitude_shape_mismatch():
    with pytest.raises(ValueError):
        gradient_magnitude(np.zeros((3, 3)), np.zeros((4, 4)))


def test_laplacian_signed_float():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    L = laplacian(A, variant="8")
    assert L.dtype == np.float64


def test_laplacian_variants():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    L4 = laplacian(A, variant="4")
    L8 = laplacian(A, variant="8")
    assert L4.shape == L8.shape


def test_laplacian_invalid_variant():
    with pytest.raises(ValueError):
        laplacian(np.zeros((5, 5)), variant="9")


def test_laplacian_uint8(sample_gray):
    out = laplacian_uint8(sample_gray, variant="8")
    assert out.dtype == np.uint8


def test_canny_binary_output(sample_gray):
    edges = canny(sample_gray, low_threshold=50, high_threshold=150)
    assert edges.shape == sample_gray.shape
    unique = np.unique(edges)
    assert set(unique).issubset({0, 255})


def test_canny_invalid_thresholds(sample_gray):
    with pytest.raises(ValueError):
        canny(sample_gray, low_threshold=200, high_threshold=50)


def test_edge_density():
    E = np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8)
    assert edge_density(E) == pytest.approx(2 / 6)


def test_edge_density_empty():
    assert edge_density(np.zeros((0, 0), dtype=np.uint8)) == 0.0