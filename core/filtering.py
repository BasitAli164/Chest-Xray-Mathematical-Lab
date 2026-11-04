"""Named filters: mean, Gaussian, median.

All filters preserve shape and return uint8 images.

Public API
----------
- mean_filter(image, kernel_size=3)
- gaussian_filter(image, kernel_size=5, sigma=1.0)
- median_filter(image, kernel_size=3)
- gaussian_kernel(kernel_size, sigma)  (returns the actual kernel matrix)
"""

from __future__ import annotations

import cv2
import numpy as np

from core.convolution import convolve_manual, validate_kernel
from core.image_preprocessing import ensure_grayscale, ensure_uint8


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------
def mean_kernel(kernel_size: int = 3) -> np.ndarray:
    """Uniform 1/(k^2) kernel of shape (k, k)."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be positive odd, got {kernel_size}."
        )
    return np.full(
        (kernel_size, kernel_size),
        1.0 / (kernel_size * kernel_size),
        dtype=np.float64,
    )


def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """2D Gaussian kernel normalized to sum 1.

    Formula:
        G(x, y) = exp(-(x^2 + y^2) / (2*sigma^2))
        K(x, y) = G(x, y) / sum(G)
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be positive odd, got {kernel_size}."
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    half = kernel_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


# ---------------------------------------------------------------------------
# Mean filter
# ---------------------------------------------------------------------------
def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Mean filter via manual convolution with a uniform kernel."""
    img = ensure_uint8(ensure_grayscale(image))
    K = mean_kernel(kernel_size)
    out = convolve_manual(img, K, padding="reflect")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gaussian filter
# ---------------------------------------------------------------------------
def gaussian_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Gaussian filter via OpenCV (matches manual convolution closely)."""
    img = ensure_uint8(ensure_grayscale(image))
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be positive odd, got {kernel_size}."
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    out = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------
def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter via OpenCV (non-linear; no linear kernel exists)."""
    img = ensure_uint8(ensure_grayscale(image))
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be positive odd, got {kernel_size}."
        )
    out = cv2.medianBlur(img, kernel_size)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Validation passthrough (import-friendly)
# ---------------------------------------------------------------------------
__all__ = [
    "mean_kernel",
    "gaussian_kernel",
    "mean_filter",
    "gaussian_filter",
    "median_filter",
    "validate_kernel",
]