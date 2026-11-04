"""Reusable numerical helpers used across the mathematical engine.

Everything here works on NumPy arrays and returns plain Python types so
the results can be JSON-serialized into traces.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------
def safe_stats(A: np.ndarray) -> dict:
    """Mean / median / std / variance / min / max / sum / size.

    Handles empty arrays by returning None for undefined entries.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a NumPy array.")
    if A.size == 0:
        return {
            "size": 0,
            "mean": None,
            "median": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "sum": 0.0,
        }
    return {
        "size": int(A.size),
        "mean": float(A.mean()),
        "median": float(np.median(A)),
        "std": float(A.std()),
        "variance": float(A.var()),
        "min": float(A.min()),
        "max": float(A.max()),
        "sum": float(A.sum()),
    }


def difference_norm(
    A: np.ndarray,
    B: np.ndarray,
    ord: str = "fro",
) -> float:
    """||A - B||_ord (default Frobenius). Shape mismatch raises."""
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A{A.shape} vs B{B.shape}."
        )
    return float(np.linalg.norm(A - B, ord=ord))


def maximum_abs_difference(A: np.ndarray, B: np.ndarray) -> float:
    """max |A[i, j] - B[i, j]| across all entries."""
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A{A.shape} vs B{B.shape}."
        )
    return float(np.max(np.abs(A.astype(np.float64) - B.astype(np.float64))))


def relative_error(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_F / ||B||_F. Returns None when ||B||_F == 0."""
    if A.shape != B.shape:
        raise ValueError(
            f"Shape mismatch: A{A.shape} vs B{B.shape}."
        )
    denom = float(np.linalg.norm(B, ord="fro"))
    if denom == 0.0:
        return None
    return float(np.linalg.norm(A - B, ord="fro") / denom)


# ---------------------------------------------------------------------------
# Matrix validation helpers (used by tests and computation_validator)
# ---------------------------------------------------------------------------
def matrix_multiply_check(A: np.ndarray, B: np.ndarray) -> dict:
    """Compare matrix_multiply_manual vs matrix_multiply_numpy."""
    from core.matrix_operations import (
        matrix_multiply_manual,
        matrix_multiply_numpy,
    )
    C_manual = matrix_multiply_manual(A, B)
    C_np = matrix_multiply_numpy(A, B)
    return {
        "match": bool(np.allclose(C_manual, C_np, rtol=1e-9, atol=1e-9)),
        "max_abs_diff": maximum_abs_difference(C_manual, C_np),
        "output_shape": list(C_np.shape),
    }