"""Reusable preprocessing helpers used by every downstream module.

Public API
----------
- ensure_uint8()          -> np.ndarray (uint8, C-contiguous)
- ensure_grayscale()      -> np.ndarray (2D)
- resize_max_dimension()  -> np.ndarray (downscale only)
- clip_to_range()         -> np.ndarray (intensity clamping)
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Guarantee a C-contiguous uint8 array.

    - uint8 already    : returned as-is (C-contiguous copy)
    - float / int      : clipped to [0, 255] then rounded

    Parameters
    ----------
    image : np.ndarray
        Input array of any numeric dtype.

    Returns
    -------
    np.ndarray
        C-contiguous uint8 array with the same shape.
    """
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    clipped = np.clip(image, 0, 255)
    return np.ascontiguousarray(np.round(clipped).astype(np.uint8))


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Guarantee a 2D grayscale array.

    - 2D already            : returned unchanged
    - (H, W, 1)             : squeezed to (H, W)
    - (H, W, C) with C > 1  : raises ValueError

    Parameters
    ----------
    image : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        2D array (same dtype as input).

    Raises
    ------
    ValueError
        If a color image (C > 1) is passed; caller must convert first.
    """
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    raise ValueError(
        f"Expected grayscale array, got shape {image.shape}. "
        "Call core.image_loader.to_grayscale() first."
    )


def resize_max_dimension(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale (never upscale) so the longer side is <= max_dim.

    Bilinear interpolation via PIL keeps the implementation dependency-light
    while preserving dtype.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image array.
    max_dim : int
        Maximum allowed length of the longer side (pixels).

    Returns
    -------
    np.ndarray
        - If already within the limit: the *same* object is returned.
        - Otherwise: a resized copy with the same dtype.
    """
    if max_dim <= 0:
        raise ValueError("max_dim must be positive.")

    h, w = image.shape[:2]

    if max(h, w) <= max_dim:
        return image

    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    pil = Image.fromarray(image)
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.array(pil)


def clip_to_range(
    image: np.ndarray,
    lo: float = 0.0,
    hi: float = 255.0,
) -> np.ndarray:
    """Clamp values into [lo, hi] without changing dtype.

    Parameters
    ----------
    image : np.ndarray
        Input array (float or int).
    lo, hi : float
        Inclusive bounds.

    Returns
    -------
    np.ndarray
        New array with values clamped to [lo, hi], dtype preserved
        (uint8 inputs are returned as uint8).
    """
    if lo > hi:
        raise ValueError(f"lo ({lo}) must be <= hi ({hi}).")

    out = np.clip(image, lo, hi)

    if image.dtype == np.uint8:
        return out.astype(np.uint8, copy=False)

    return out