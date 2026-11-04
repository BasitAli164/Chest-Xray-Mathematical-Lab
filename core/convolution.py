"""Convolution — both manual (educational) and optimized implementations.

The manual implementation uses explicit Python loops so every
multiplication and summation can be traced. The optimized implementation
uses SciPy/OpenCV under the hood, and MUST produce identical output
(within rounding) for integer inputs.

Convolution definition (used in this project):

    O(i, j) = sum_u sum_v I(i-u, j-v) * K(u, v)

This is the *cross-correlation* form common in image processing (as
opposed to the flipped-kernel mathematical convolution). For symmetric
kernels (mean, Gaussian, Sobel magnitudes) the two forms coincide.

Public API
----------
- validate_kernel(kernel)
- convolve_manual(image, kernel, padding="reflect")
- convolve_scipy(image, kernel)
- convolve_opencv(image, kernel)
- pad_image(image, pad_h, pad_w, mode)
- compare_implementations(image, kernel)
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy.signal import convolve2d

from core.image_preprocessing import ensure_grayscale

PaddingMode = Literal["zero", "reflect", "replicate", "wrap"]


# ---------------------------------------------------------------------------
# Kernel validation
# ---------------------------------------------------------------------------
def validate_kernel(kernel: np.ndarray) -> None:
    """Ensure kernel is a 2D, non-empty, odd-dimension, finite array.

    Raises
    ------
    ValueError
        On any violation.
    """
    if not isinstance(kernel, np.ndarray):
        raise ValueError("Kernel must be a NumPy array.")
    if kernel.ndim != 2:
        raise ValueError(
            f"Kernel must be 2D, got shape {kernel.shape}."
        )
    if kernel.size == 0:
        raise ValueError("Kernel is empty.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(
            f"Kernel dimensions must be odd, got {kernel.shape}."
        )
    if not np.all(np.isfinite(kernel)):
        raise ValueError("Kernel contains NaN or Inf.")


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------
def pad_image(
    image: np.ndarray,
    pad_h: int,
    pad_w: int,
    mode: PaddingMode = "reflect",
) -> np.ndarray:
    """Pad the image by (pad_h, pad_w) on each side.

    Parameters
    ----------
    image : np.ndarray
        2D input array.
    pad_h, pad_w : int
        Number of rows/columns to add on each side.
    mode : {"zero", "reflect", "replicate", "wrap"}
        Boundary handling.

    Returns
    -------
    np.ndarray
        Padded image with shape (H + 2*pad_h, W + 2*pad_w).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}.")
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Padding must be >= 0, got ({pad_h}, {pad_w}).")
    if pad_h == 0 and pad_w == 0:
        return image.copy()

    mode_map = {
        "zero": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "wrap": "wrap",
    }
    if mode not in mode_map:
        raise ValueError(
            f"Unsupported padding mode '{mode}'. "
            f"Choose from {list(mode_map)}."
        )

    return np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode=mode_map[mode],
    )


# ---------------------------------------------------------------------------
# Manual convolution (educational)
# ---------------------------------------------------------------------------
def convolve_manual(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: PaddingMode = "reflect",
) -> np.ndarray:
    """Convolution with explicit loops — educational reference implementation.

    For an odd-sized kernel of shape (kh, kw), pad = (kh//2, kw//2).
    Output has the same shape as the input (padding is stripped after
    convolution).

    Parameters
    ----------
    image : np.ndarray
        2D input (any numeric dtype).
    kernel : np.ndarray
        Odd-sized 2D kernel.
    padding : PaddingMode
        Boundary handling.

    Returns
    -------
    np.ndarray
        float64 array of the same shape as input.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}.")
    validate_kernel(kernel)

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    src = image.astype(np.float64)
    padded = pad_image(src, pad_h, pad_w, mode=padding)
    H, W = src.shape
    out = np.zeros((H, W), dtype=np.float64)
    K = kernel.astype(np.float64)

    # Explicit loops — for educational tracing only.
    for i in range(H):
        for j in range(W):
            window = padded[i:i + kh, j:j + kw]
            acc = 0.0
            for u in range(kh):
                for v in range(kw):
                    acc += window[u, v] * K[u, v]
            out[i, j] = acc

    return out


# ---------------------------------------------------------------------------
# Optimized implementations
# ---------------------------------------------------------------------------
def convolve_scipy(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution using scipy.signal.convolve2d (boundary='symm')."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}.")
    validate_kernel(kernel)

    src = image.astype(np.float64)
    K = kernel.astype(np.float64)
    return convolve2d(src, K, mode="same", boundary="symm")


def convolve_opencv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution using OpenCV's filter2D (BORDER_REFLECT_101)."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}.")
    validate_kernel(kernel)

    src = image.astype(np.float64)
    K = kernel.astype(np.float64)
    return cv2.filter2D(src, ddepth=-1, kernel=K, borderType=cv2.BORDER_REFLECT_101)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
def compare_implementations(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: PaddingMode = "reflect",
) -> dict:
    """Run manual + scipy + opencv and report numerical differences.

    Returns
    -------
    dict with keys:
        "manual_shape", "scipy_shape", "opencv_shape",
        "manual_vs_scipy_max_abs_diff",
        "manual_vs_opencv_max_abs_diff",
        "match_scipy", "match_opencv"
    """
    manual = convolve_manual(image, kernel, padding=padding)
    sci = convolve_scipy(image, kernel)
    ocv = convolve_opencv(image, kernel)

    d_scipy = float(np.max(np.abs(manual - sci)))
    d_ocv = float(np.max(np.abs(manual - ocv)))

    return {
        "manual_shape": list(manual.shape),
        "scipy_shape": list(sci.shape),
        "opencv_shape": list(ocv.shape),
        "manual_vs_scipy_max_abs_diff": d_scipy,
        "manual_vs_opencv_max_abs_diff": d_ocv,
        "match_scipy": bool(np.allclose(manual, sci, atol=1e-9)),
        "match_opencv": bool(np.allclose(manual, ocv, atol=1e-6)),
    }