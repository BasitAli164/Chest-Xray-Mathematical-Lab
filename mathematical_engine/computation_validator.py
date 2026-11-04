"""Validators used at the entrance of every computation.

Each validator returns a dict:
    {"ok": bool, "warnings": [...], "errors": [...]}

The core layer uses `.require()` for hard failures and `.soft()` for
soft warnings (e.g. "output range exceeded uint8").
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_array(
    A: Any,
    name: str = "A",
    require_2d: bool = True,
    require_finite: bool = True,
) -> dict[str, Any]:
    """Validate that A is a NumPy array with sane properties."""
    warnings: list[str] = []
    errors: list[str] = []

    if not isinstance(A, np.ndarray):
        errors.append(f"{name} must be a NumPy array, got {type(A).__name__}.")
        return {"ok": False, "warnings": warnings, "errors": errors}

    if A.size == 0:
        errors.append(f"{name} is empty (size 0).")

    if require_2d and A.ndim != 2:
        errors.append(f"{name} must be 2D, got shape {A.shape}.")

    if require_finite and A.dtype.kind in ("f", "c"):
        if not np.all(np.isfinite(A)):
            errors.append(f"{name} contains NaN or Inf.")

    if A.dtype.kind not in ("u", "i", "f"):
        warnings.append(
            f"{name} has non-numeric dtype {A.dtype}."
        )

    return {"ok": len(errors) == 0, "warnings": warnings, "errors": errors}


def validate_same_shape(
    A: np.ndarray,
    B: np.ndarray,
    names: tuple[str, str] = ("A", "B"),
) -> dict[str, Any]:
    """Validate that two arrays have identical shape."""
    if A.shape == B.shape:
        return {"ok": True, "warnings": [], "errors": []}
    return {
        "ok": False,
        "warnings": [],
        "errors": [
            f"{names[0]}{A.shape} and {names[1]}{B.shape} must have the same shape."
        ],
    }


def validate_multiply_shapes(
    A: np.ndarray,
    B: np.ndarray,
) -> dict[str, Any]:
    """Validate A @ B is possible: A.shape[1] == B.shape[0]."""
    if A.ndim != 2 or B.ndim != 2:
        return {
            "ok": False,
            "warnings": [],
            "errors": ["Both A and B must be 2D for matrix multiply."],
        }
    if A.shape[1] == B.shape[0]:
        return {"ok": True, "warnings": [], "errors": []}
    return {
        "ok": False,
        "warnings": [],
        "errors": [
            f"Inner dimensions mismatch: A{A.shape} @ B{B.shape}."
        ],
    }


def validate_intensity_range(
    A: np.ndarray,
    lo: float = 0.0,
    hi: float = 255.0,
) -> dict[str, Any]:
    """Warn if values fall outside [lo, hi]."""
    warnings: list[str] = []
    if A.size == 0:
        return {"ok": True, "warnings": [], "errors": []}
    a_min = float(A.min())
    a_max = float(A.max())
    if a_min < lo or a_max > hi:
        warnings.append(
            f"Values outside [{lo}, {hi}]: actual [{a_min}, {a_max}]."
        )
    return {"ok": True, "warnings": warnings, "errors": []}


def require(result: dict[str, Any]) -> None:
    """Raise ValueError on any error in the validation result."""
    if not result["ok"]:
        joined = " | ".join(result["errors"]) or "validation failed"
        raise ValueError(joined)