"""Edge detection: Sobel X/Y, gradient magnitude, Laplacian, Canny.

All kernels are exposed as constants so the UI can display them, and
manual convolution is used for Sobel X/Y and Laplacian to allow
step-by-step tracing.

Public API
----------
- SOBEL_X, SOBEL_Y, LAPLACIAN_4, LAPLACIAN_8  (kernel constants)
- sobel_x(image)
- sobel_y(image)
- gradient_magnitude(gx, gy)
- laplacian(image, variant="8")
- canny(image, low, high)
- edge_density(edge_map, threshold=0)
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from core.convolution import convolve_manual
from core.image_preprocessing import ensure_grayscale, ensure_uint8


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
SOBEL_X: np.ndarray = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=np.float64,
)

SOBEL_Y: np.ndarray = np.array(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]],
    dtype=np.float64,
)

LAPLACIAN_4: np.ndarray = np.array(
    [[0,  1, 0],
     [1, -4, 1],
     [0,  1, 0]],
    dtype=np.float64,
)

LAPLACIAN_8: np.ndarray = np.array(
    [[1,  1,  1],
     [1, -8,  1],
     [1,  1,  1]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Sobel
# ---------------------------------------------------------------------------
def _sobel_signed(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Shared implementation: returns the *signed* float64 response."""
    img = ensure_uint8(ensure_grayscale(image))
    out = convolve_manual(img, kernel, padding="reflect")
    return out


def sobel_x(image: np.ndarray) -> np.ndarray:
    """Signed Gx response (float64). Positive = intensity increasing to the right."""
    return _sobel_signed(image, SOBEL_X)


def sobel_y(image: np.ndarray) -> np.ndarray:
    """Signed Gy response (float64). Positive = intensity increasing downward."""
    return _sobel_signed(image, SOBEL_Y)


def sobel_uint8(image: np.ndarray, axis: Literal["x", "y"] = "x") -> np.ndarray:
    """Absolute-value version of Sobel response clipped to uint8.

    Useful for display. For analysis, prefer the signed float64 versions.
    """
    signed = sobel_x(image) if axis == "x" else sobel_y(image)
    return np.clip(np.abs(signed), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gradient magnitude
# ---------------------------------------------------------------------------
def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Magnitude = sqrt(gx^2 + gy^2). Returns float64."""
    if gx.shape != gy.shape:
        raise ValueError(
            f"gx and gy must have same shape, got {gx.shape} vs {gy.shape}."
        )
    return np.sqrt(gx ** 2 + gy ** 2)


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------
def laplacian(
    image: np.ndarray,
    variant: Literal["4", "8"] = "8",
) -> np.ndarray:
    """Signed Laplacian response (float64).

    variant="4" -> LAPLACIAN_4 kernel
    variant="8" -> LAPLACIAN_8 kernel
    """
    if variant == "4":
        kernel = LAPLACIAN_4
    elif variant == "8":
        kernel = LAPLACIAN_8
    else:
        raise ValueError(f"variant must be '4' or '8', got {variant!r}.")
    img = ensure_uint8(ensure_grayscale(image))
    return convolve_manual(img, kernel, padding="reflect")


def laplacian_uint8(image: np.ndarray, variant: Literal["4", "8"] = "8") -> np.ndarray:
    """Absolute-value version clipped to uint8 (for display)."""
    signed = laplacian(image, variant=variant)
    return np.clip(np.abs(signed), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Canny (OpenCV — internal stages not individually exposed)
# ---------------------------------------------------------------------------
def canny(
    image: np.ndarray,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
) -> np.ndarray:
    """Canny edge detection via OpenCV.

    Canny runs multi-stage processing (Gaussian smoothing, gradient,
    non-max suppression, double thresholding, hysteresis). Only the
    final binary edge map is exposed to the caller — the individual
    internal stages are not traced because OpenCV does not expose them.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    low_threshold : float
        Lower hysteresis threshold.
    high_threshold : float
        Upper hysteresis threshold.

    Returns
    -------
    np.ndarray
        uint8 binary map (0 or 255) with same shape.
    """
    img = ensure_uint8(ensure_grayscale(image))
    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Canny thresholds must be >= 0.")
    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be < "
            f"high_threshold ({high_threshold})."
        )
    return cv2.Canny(img, low_threshold, high_threshold)


# ---------------------------------------------------------------------------
# Edge density
# ---------------------------------------------------------------------------
def edge_density(edge_map: np.ndarray, threshold: int = 0) -> float:
    """Fraction of pixels classified as edges.

    Parameters
    ----------
    edge_map : np.ndarray
        2D edge map (uint8 or float).
    threshold : int
        Pixels with value > threshold are counted as edges.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if edge_map.ndim != 2:
        raise ValueError(
            f"edge_map must be 2D, got shape {edge_map.shape}."
        )
    if edge_map.size == 0:
        return 0.0
    return float(np.mean(edge_map > threshold))