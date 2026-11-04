"""Tests for core.matrix_operations."""

from __future__ import annotations

import numpy as np
import pytest

from core.matrix_operations import (
    add_matrices,
    column_statistics,
    diagonal,
    elementwise_multiply,
    extract_block,
    extract_column,
    extract_row,
    frobenius_norm_manual,
    frobenius_norm_numpy,
    matrix_multiply_manual,
    matrix_multiply_numpy,
    matrix_statistics,
    row_statistics,
    scalar_multiply,
    subtract_matrices,
    trace_matrix,
    transpose_matrix,
)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------
def test_transpose_shape_and_values():
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    At = transpose_matrix(A)
    assert At.shape == (3, 2)
    assert At[0, 0] == 1
    assert At[0, 1] == 4
    assert At[2, 1] == 6


def test_transpose_rejects_1d():
    with pytest.raises(ValueError):
        transpose_matrix(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# Add / subtract / scalar
# ---------------------------------------------------------------------------
def test_add_matrices():
    A = np.array([[1, 2], [3, 4]], dtype=np.int32)
    B = np.array([[10, 20], [30, 40]], dtype=np.int32)
    C = add_matrices(A, B)
    assert np.array_equal(C, np.array([[11, 22], [33, 44]]))


def test_add_rejects_mismatch():
    A = np.zeros((2, 2))
    B = np.zeros((2, 3))
    with pytest.raises(ValueError, match="Shape mismatch"):
        add_matrices(A, B)


def test_subtract_matrices():
    A = np.array([[5, 6], [7, 8]], dtype=np.int32)
    B = np.array([[1, 2], [3, 4]], dtype=np.int32)
    C = subtract_matrices(A, B)
    assert np.array_equal(C, np.array([[4, 4], [4, 4]]))


def test_scalar_multiply():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    C = scalar_multiply(A, 2.5)
    assert np.allclose(C, np.array([[2.5, 5], [7.5, 10]]))


# ---------------------------------------------------------------------------
# Element-wise
# ---------------------------------------------------------------------------
def test_elementwise_multiply():
    A = np.array([[1, 2], [3, 4]], dtype=np.int32)
    B = np.array([[5, 6], [7, 8]], dtype=np.int32)
    C = elementwise_multiply(A, B)
    assert np.array_equal(C, np.array([[5, 12], [21, 32]]))


# ---------------------------------------------------------------------------
# Matrix product — manual vs numpy
# ---------------------------------------------------------------------------
def test_matrix_multiply_manual_vs_numpy():
    rng = np.random.default_rng(42)
    A = rng.integers(0, 10, size=(3, 4)).astype(np.float64)
    B = rng.integers(0, 10, size=(4, 2)).astype(np.float64)
    Cm = matrix_multiply_manual(A, B)
    Cn = matrix_multiply_numpy(A, B)
    assert Cm.shape == Cn.shape == (3, 2)
    assert np.allclose(Cm, Cn)


def test_matrix_multiply_shape_mismatch():
    A = np.zeros((2, 3))
    B = np.zeros((4, 5))
    with pytest.raises(ValueError, match="inner dimensions"):
        matrix_multiply_numpy(A, B)
    with pytest.raises(ValueError, match="inner dimensions"):
        matrix_multiply_manual(A, B)


# ---------------------------------------------------------------------------
# Row / column / block
# ---------------------------------------------------------------------------
def test_extract_row_and_column(sample_gray):
    r = extract_row(sample_gray, 0)
    c = extract_column(sample_gray, 0)
    assert r.shape == (64,)
    assert c.shape == (64,)
    assert np.array_equal(r, sample_gray[0, :])
    assert np.array_equal(c, sample_gray[:, 0])


def test_extract_row_out_of_range(sample_gray):
    with pytest.raises(IndexError):
        extract_row(sample_gray, 999)


def test_extract_block(sample_gray):
    b = extract_block(sample_gray, 0, 0, 4, 4)
    assert b.shape == (4, 4)
    assert np.array_equal(b, sample_gray[0:4, 0:4])


def test_extract_block_out_of_range(sample_gray):
    with pytest.raises(ValueError):
        extract_block(sample_gray, 60, 60, 10, 10)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_row_statistics(sample_gray):
    s = row_statistics(sample_gray, 0)
    assert s["length"] == 64
    assert s["min"] == int(sample_gray[0].min())
    assert s["max"] == int(sample_gray[0].max())


def test_column_statistics(sample_gray):
    s = column_statistics(sample_gray, 5)
    assert s["length"] == 64
    assert s["mean"] == pytest.approx(float(sample_gray[:, 5].mean()))


def test_matrix_statistics(sample_gray):
    s = matrix_statistics(sample_gray)
    assert s["shape"] == [64, 64]
    assert s["dtype"] == "uint8"
    assert s["mean"] == pytest.approx(float(sample_gray.mean()))
    assert s["variance"] == pytest.approx(float(sample_gray.var()))


# ---------------------------------------------------------------------------
# Diagonal / trace / norms
# ---------------------------------------------------------------------------
def test_diagonal_and_trace():
    A = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert np.array_equal(diagonal(A), np.array([1, 4]))
    assert trace_matrix(A) == 5


def test_diagonal_requires_square():
    with pytest.raises(ValueError):
        diagonal(np.zeros((2, 3)))


def test_frobenius_manual_vs_numpy():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    assert frobenius_norm_manual(A) == pytest.approx(
        frobenius_norm_numpy(A), rel=1e-12
    )