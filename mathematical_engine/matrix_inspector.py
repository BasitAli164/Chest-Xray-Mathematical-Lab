"""Inspector utilities to pull human-readable previews from matrices.

These helpers are used by the Mathematical Workspace to display
neighborhoods, rows, columns, and numeric summaries without dumping the
whole image into the frontend.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def neighborhood(
    A: np.ndarray,
    row: int,
    col: int,
    size: int = 3,
) -> np.ndarray:
    """Return the size x size neighborhood centered on (row, col).

    Parameters
    ----------
    A : np.ndarray
        2D matrix.
    row, col : int
        Center coordinates.
    size : int
        Odd window size (3, 5, ...).

    Returns
    -------
    np.ndarray
        The clipped neighborhood. If the neighborhood would extend past
        the borders, it is clipped — the caller sees the actual clipped
        shape.

    Raises
    ------
    ValueError
        If A is not 2D or size is not a positive odd integer.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}.")
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size}.")

    half = size // 2
    r0 = max(0, row - half)
    r1 = min(A.shape[0], row + half + 1)
    c0 = max(0, col - half)
    c1 = min(A.shape[1], col + half + 1)
    return A[r0:r1, c0:c1].copy()


def row_preview(
    A: np.ndarray,
    row: int,
    max_values: int = 20,
) -> dict[str, Any]:
    """Preview of a single row: length, first/last values, stats."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}.")
    if not (0 <= row < A.shape[0]):
        raise IndexError(f"Row {row} out of range for A{A.shape}.")
    r = A[row, :]
    n = min(max_values, r.size)
    return {
        "row_index": int(row),
        "length": int(r.size),
        "first_values": [float(v) for v in r[:n]],
        "stats": {
            "min": float(r.min()),
            "max": float(r.max()),
            "mean": float(r.mean()),
            "std": float(r.std()),
        },
    }


def column_preview(
    A: np.ndarray,
    col: int,
    max_values: int = 20,
) -> dict[str, Any]:
    """Preview of a single column."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}.")
    if not (0 <= col < A.shape[1]):
        raise IndexError(f"Column {col} out of range for A{A.shape}.")
    c = A[:, col]
    n = min(max_values, c.size)
    return {
        "column_index": int(col),
        "length": int(c.size),
        "first_values": [float(v) for v in c[:n]],
        "stats": {
            "min": float(c.min()),
            "max": float(c.max()),
            "mean": float(c.mean()),
            "std": float(c.std()),
        },
    }


def block_preview(
    A: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> dict[str, Any]:
    """Preview of an h x w block starting at (r0, c0)."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}.")
    if h <= 0 or w <= 0:
        raise ValueError(f"Block size must be positive, got {h}x{w}.")
    r1 = min(A.shape[0], r0 + h)
    c1 = min(A.shape[1], c0 + w)
    if r0 < 0 or c0 < 0 or r0 >= r1 or c0 >= c1:
        raise ValueError(
            f"Invalid block ({r0}, {c0}, {h}, {w}) for A{A.shape}."
        )
    block = A[r0:r1, c0:c1]
    return {
        "origin": [int(r0), int(c0)],
        "shape": list(block.shape),
        "values": block.tolist(),
    }


def center_pixel(A: np.ndarray) -> tuple[int, int]:
    """Return (row, col) of the center pixel."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}.")
    return A.shape[0] // 2, A.shape[1] // 2