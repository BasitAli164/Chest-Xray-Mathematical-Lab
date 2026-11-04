"""Point-wise intensity transformations.

All transformations map an input array to a new array of the same shape.
Outputs are float64 in [0, 255] unless documented otherwise; downstream
code (Phase 3 enhancement) is responsible for uint8 conversion if needed.

Public API
----------
- identity_transform(image)
- linear_scaling(image, gain, offset)
- negative_transform(image)
- logarithmic_transform(image, c=1.0)
- gamma_transform(image, gamma)
- contrast_stretch(image)      (min-max normalization to [0, 255])
"""

from __future__ import annotations

import numpy as np


def _as_float(image: np.ndarray) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D image, got shape {image.shape}."
        )
    return image.astype(np.float64)


def identity_transform(image: np.ndarray) -> np.ndarray:
    """s = r — no-op intensity mapping.

    Useful as a baseline in the Mathematical Workspace.
    """
    return _as_float(image).copy()


def linear_scaling(
    image: np.ndarray,
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """s = gain * r + offset, clipped to [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Input image.
    gain : float
        Multiplicative factor (contrast-like effect).
    offset : float
        Additive factor (brightness-like effect).

    Returns
    -------
    np.ndarray (float64)
        Values clipped to [0, 255].
    """
    arr = _as_float(image)
    out = gain * arr + offset
    return np.clip(out, 0.0, 255.0)


def negative_transform(image: np.ndarray) -> np.ndarray:
    """s = 255 - r."""
    arr = _as_float(image)
    return 255.0 - arr


def logarithmic_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """s = c * log(1 + r), clipped to [0, 255].

    Compresses bright regions, expands dark ones — classic for X-ray
    visualization.
    """
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}.")
    arr = _as_float(image)
    out = c * np.log1p(arr)
    return np.clip(out, 0.0, 255.0)


def gamma_transform(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """s = 255 * (r / 255) ** gamma.

    gamma < 1  -> brightens
    gamma > 1  -> darkens
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}.")
    arr = _as_float(image)
    out = 255.0 * np.power(arr / 255.0, gamma)
    return np.clip(out, 0.0, 255.0)


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    """Min-max normalization to full [0, 255] range.

    Formula:
        s = (r - r_min) / (r_max - r_min) * 255

    If r_max == r_min (constant image), returns zeros.
    """
    arr = _as_float(image)
    r_min = arr.min()
    r_max = arr.max()
    if r_max == r_min:
        return np.zeros_like(arr)
    return (arr - r_min) * (255.0 / (r_max - r_min))