"""Matrix operations on image arrays.

This module is the linear-algebra workhorse of the application.

Design contract
---------------
- Every function accepts NumPy arrays and returns NumPy arrays (or plain
  Python scalars). No Streamlit, no global state.
- "Manual" implementations (loops) are provided alongside optimized NumPy
  versions where the educational value is high, so the mathematical engine
  can compare them.
- All functions validate input shapes and raise ValueError on mismatch.

Public API
----------
- transpose_matrix(A)
- add_matrices(A, B)
- subtract_matrices(A, B)
- scalar_multiply(A, k)
- elementwise_multiply(A, B)
- matrix_multiply_manual(A, B)
- matrix_multiply_numpy(A, B)
- extract_row(A, i)
- extract_column(A, j)
- extract_block(A, r0, c0, h, w)
- row_statistics(A, i)
- column_statistics(A, j)
- matrix_statistics(A)
- diagonal(A)
- trace_matrix(A)
- frobenius_norm_manual(A)
- frobenius_norm_numpy(A)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------
def _require_2d(A: np.ndarray, name: str = "A") -> None:
    if A.ndim != 2:
        raise ValueError(
            f"{name} must be 2D, got shape {A.shape}."
        )


def _require_same_shape(A: np.ndarray, B: np.ndarray) -> None:
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A{A.shape} vs B{B.shape}."
        )


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------
def transpose_matrix(A: np.ndarray) -> np.ndarray:
    """Return A^T.

    For an (m, n) matrix, the transpose is (n, m) with:
        A^T[j, i] = A[i, j]

    Parameters
    ----------
    A : np.ndarray
        2D input array.

    Returns
    -------
    np.ndarray
        Transposed array (a view-based copy via np.transpose).
    """
    _require_2d(A)
    return np.ascontiguousarray(A.T)


# ---------------------------------------------------------------------------
# Add / subtract / scalar
# ---------------------------------------------------------------------------
def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Element-wise addition: C[i, j] = A[i, j] + B[i, j]."""
    _require_2d(A)
    _require_2d(B)
    _require_same_shape(A, B)
    return A + B


def subtract_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Element-wise subtraction: C[i, j] = A[i, j] - B[i, j]."""
    _require_2d(A)
    _require_2d(B)
    _require_same_shape(A, B)
    return A - B


def scalar_multiply(A: np.ndarray, k: float) -> np.ndarray:
    """Scalar multiplication: C[i, j] = k * A[i, j]."""
    _require_2d(A)
    return A * k


# ---------------------------------------------------------------------------
# Element-wise (Hadamard) and matrix product
# ---------------------------------------------------------------------------
def elementwise_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Hadamard product: C[i, j] = A[i, j] * B[i, j]."""
    _require_2d(A)
    _require_2d(B)
    _require_same_shape(A, B)
    return A * B


def matrix_multiply_manual(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix product AB computed with explicit loops.

    Educational reference implementation. For an (m, n) A and (n, p) B:
        C[i, j] = sum_{k=0}^{n-1} A[i, k] * B[k, j]

    Slow (O(m*n*p) Python-level) — intended for small verification cases.
    """
    _require_2d(A)
    _require_2d(B)
    m, n = A.shape
    n_b, p = B.shape
    if n != n_b:
        raise ValueError(
            f"Cannot multiply A{A.shape} @ B{B.shape}: "
            f"inner dimensions {n} != {n_b}."
        )

    C = np.zeros((m, p), dtype=np.result_type(A.dtype, B.dtype))
    for i in range(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def matrix_multiply_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix product AB using NumPy's optimized BLAS-backed @ operator."""
    _require_2d(A)
    _require_2d(B)
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Cannot multiply A{A.shape} @ B{B.shape}: "
            f"inner dimensions {A.shape[1]} != {B.shape[0]}."
        )
    return A @ B


# ---------------------------------------------------------------------------
# Row / column / block extraction
# ---------------------------------------------------------------------------
def extract_row(A: np.ndarray, i: int) -> np.ndarray:
    """Return A[i, :] as a 1D copy."""
    _require_2d(A)
    if not (0 <= i < A.shape[0]):
        raise IndexError(
            f"Row index {i} out of range for A{A.shape}."
        )
    return A[i, :].copy()


def extract_column(A: np.ndarray, j: int) -> np.ndarray:
    """Return A[:, j] as a 1D copy."""
    _require_2d(A)
    if not (0 <= j < A.shape[1]):
        raise IndexError(
            f"Column index {j} out of range for A{A.shape}."
        )
    return A[:, j].copy()


def extract_block(
    A: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Return the h×w block starting at (r0, c0).

    Validates that [r0, r0+h) and [c0, c0+w) lie fully inside A.
    """
    _require_2d(A)
    m, n = A.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Block size must be positive, got {h}x{w}.")
    if r0 < 0 or c0 < 0:
        raise ValueError(f"Negative block origin ({r0}, {c0}).")
    if r0 + h > m or c0 + w > n:
        raise ValueError(
            f"Block ({h}x{w} at {r0},{c0}) exceeds A{A.shape}."
        )
    return A[r0:r0 + h, c0:c0 + w].copy()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def row_statistics(A: np.ndarray, i: int) -> dict:
    """Statistics of row i."""
    row = extract_row(A, i)
    return {
        "index": int(i),
        "length": int(row.size),
        "min": float(row.min()),
        "max": float(row.max()),
        "mean": float(row.mean()),
        "median": float(np.median(row)),
        "std": float(row.std()),
        "sum": float(row.sum()),
    }


def column_statistics(A: np.ndarray, j: int) -> dict:
    """Statistics of column j."""
    col = extract_column(A, j)
    return {
        "index": int(j),
        "length": int(col.size),
        "min": float(col.min()),
        "max": float(col.max()),
        "mean": float(col.mean()),
        "median": float(np.median(col)),
        "std": float(col.std()),
        "sum": float(col.sum()),
    }


def matrix_statistics(A: np.ndarray) -> dict:
    """Global statistics of the whole matrix."""
    _require_2d(A)
    return {
        "shape": list(A.shape),
        "dtype": str(A.dtype),
        "size": int(A.size),
        "min": float(A.min()),
        "max": float(A.max()),
        "mean": float(A.mean()),
        "median": float(np.median(A)),
        "std": float(A.std()),
        "variance": float(A.var()),
        "sum": float(A.sum()),
    }


# ---------------------------------------------------------------------------
# Trace and norms
# ---------------------------------------------------------------------------
def diagonal(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal (requires square A)."""
    _require_2d(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Diagonal requires a square matrix, got {A.shape}."
        )
    return np.diag(A).copy()


def trace_matrix(A: np.ndarray) -> float:
    """Trace = sum of the main diagonal (requires square A)."""
    return float(diagonal(A).sum())


def frobenius_norm_manual(A: np.ndarray) -> float:
    """Frobenius norm ||A||_F = sqrt(sum(A[i,j]^2)) — manual loop."""
    _require_2d(A)
    total = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = float(A[i, j])
            total += v * v
    return total ** 0.5


def frobenius_norm_numpy(A: np.ndarray) -> float:
    """Frobenius norm via NumPy (equivalent to np.linalg.norm(A, 'fro'))."""
    _require_2d(A)
    return float(np.linalg.norm(A, ord="fro"))