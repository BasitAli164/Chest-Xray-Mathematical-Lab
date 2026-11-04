"""Tests for mathematical_engine.pixel_trace."""

from __future__ import annotations

import numpy as np
import pytest

from core.convolution import convolve_manual
from core.edge_detection import SOBEL_X
from mathematical_engine.pixel_trace import (
    trace_convolution_pixel,
    trace_sobel_pixel,
)


def test_trace_identity_kernel():
    A = np.arange(25, dtype=np.float64).reshape(5, 5)
    K = np.zeros((3, 3))
    K[1, 1] = 1.0
    tr = trace_convolution_pixel(A, K, 2, 2, padding="reflect")
    assert tr["sum"] == A[2, 2]
    assert len(tr["multiplications"]) == 9


def test_trace_matches_actual_convolution():
    A = np.arange(100, dtype=np.float64).reshape(10, 10)
    K = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    out = convolve_manual(A, K, padding="reflect")
    tr = trace_convolution_pixel(A, K, 5, 5, padding="reflect")
    assert tr["result"] == pytest.approx(out[5, 5])


def test_trace_out_of_range():
    A = np.zeros((5, 5))
    K = np.ones((3, 3))
    with pytest.raises(IndexError):
        trace_convolution_pixel(A, K, 10, 10)


def test_trace_invalid_kernel_even_size():
    A = np.zeros((5, 5))
    K = np.ones((2, 2))
    with pytest.raises(ValueError):
        trace_convolution_pixel(A, K, 2, 2)


def test_trace_sobel():
    A = np.arange(100, dtype=np.float64).reshape(10, 10)
    tr = trace_sobel_pixel(A, 5, 5, axis="x")
    assert tr["axis"] == "x"
    assert tr["kernel_shape"] == [3, 3]


def test_trace_sobel_invalid_axis():
    A = np.zeros((10, 10))
    with pytest.raises(ValueError):
        trace_sobel_pixel(A, 5, 5, axis="z")


def test_trace_multiplications_count():
    A = np.ones((10, 10))
    K = np.ones((5, 5)) / 25.0
    tr = trace_convolution_pixel(A, K, 5, 5)
    assert len(tr["multiplications"]) == 25


def test_trace_kernel_stored():
    A = np.ones((10, 10))
    K = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    tr = trace_convolution_pixel(A, K, 5, 5)
    assert tr["kernel_values"] == K.tolist()