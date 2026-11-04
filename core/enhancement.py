"""Image enhancement operations.

All functions accept a 2D grayscale uint8 array and return a
2D uint8 array of the same shape. Operations preserve the [0, 255]
intensity range.

Public API
----------
- adjust_brightness(image, offset)
- adjust_contrast(image, gain)
- min_max_normalize(image)
- histogram_equalization_manual(image)
- histogram_equalization_opencv(image)
- apply_clahe(image, clip_limit, tile_grid_size)
- sharpen(image, strength)
"""

from __future__ import annotations

import cv2
import numpy as np

from core.image_preprocessing import ensure_grayscale, ensure_uint8


# ---------------------------------------------------------------------------
# Brightness
# ---------------------------------------------------------------------------
def adjust_brightness(image: np.ndarray, offset: int = 0) -> np.ndarray:
    """s = r + offset, clipped to [0, 255].

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    offset : int
        Additive brightness shift. Positive -> brighter, negative -> darker.

    Returns
    -------
    np.ndarray
        uint8 image with same shape.
    """
    img = ensure_uint8(ensure_grayscale(image))
    out = img.astype(np.int32) + int(offset)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------
def adjust_contrast(image: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """s = gain * (r - 128) + 128, clipped to [0, 255].

    Contrast is adjusted around the midpoint (128) so that brightness
    stays roughly constant.
    """
    img = ensure_uint8(ensure_grayscale(image))
    arr = img.astype(np.float64)
    out = float(gain) * (arr - 128.0) + 128.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Min-max normalization
# ---------------------------------------------------------------------------
def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Stretch intensities to full [0, 255] range.

    Formula:
        s = (r - r_min) / (r_max - r_min) * 255

    Constant image -> all zeros.
    """
    img = ensure_uint8(ensure_grayscale(image))
    arr = img.astype(np.float64)
    r_min = float(arr.min())
    r_max = float(arr.max())
    if r_max == r_min:
        return np.zeros_like(img, dtype=np.uint8)
    out = (arr - r_min) * (255.0 / (r_max - r_min))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Histogram equalization — manual
# ---------------------------------------------------------------------------
def histogram_equalization_manual(image: np.ndarray) -> np.ndarray:
    """Histogram equalization using cumulative distribution function.

    Algorithm
    ---------
    1. Compute histogram h[r] for r in [0, 255].
    2. Compute CDF: cdf[r] = sum_{k=0..r} h[k].
    3. Normalize: s[r] = round((cdf[r] - cdf_min) / (N - cdf_min) * 255).
    4. Map each pixel r -> s[r].

    For a constant image (all pixels equal), returns zeros.

    Uses explicit loops for the CDF, so the mathematical trace can
    expose intermediate values.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.

    Returns
    -------
    np.ndarray
        uint8 image with same shape.
    """
    img = ensure_uint8(ensure_grayscale(image))
    flat = img.ravel()
    n = flat.size

    # Step 1: histogram
    hist = np.zeros(256, dtype=np.int64)
    for v in flat:
        hist[int(v)] += 1

    # Step 2: cumulative sum
    cdf = np.zeros(256, dtype=np.int64)
    running = 0
    for i in range(256):
        running += int(hist[i])
        cdf[i] = running

    cdf_min = int(cdf[cdf > 0][0]) if np.any(cdf > 0) else 0
    denom = n - cdf_min
    if denom <= 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Step 3: mapping
    mapping = np.zeros(256, dtype=np.uint8)
    for r in range(256):
        mapping[r] = int(round((cdf[r] - cdf_min) / denom * 255.0))

    # Step 4: apply
    out = mapping[img]
    return out.astype(np.uint8)


def histogram_equalization_opencv(image: np.ndarray) -> np.ndarray:
    """OpenCV's optimized histogram equalization.

    Provided as a comparison baseline for the manual implementation.
    Numerical results should match closely (may differ by +/- 1 due to
    rounding convention).
    """
    img = ensure_uint8(ensure_grayscale(image))
    return cv2.equalizeHist(img)


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------
def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (OpenCV).

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    clip_limit : float
        Threshold for contrast limiting (higher -> more contrast).
    tile_grid_size : tuple[int, int]
        Grid size for local histograms.

    Returns
    -------
    np.ndarray
        uint8 image with same shape.
    """
    img = ensure_uint8(ensure_grayscale(image))
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}.")
    if len(tile_grid_size) != 2 or min(tile_grid_size) <= 0:
        raise ValueError(
            f"tile_grid_size must be 2 positive ints, got {tile_grid_size}."
        )
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(int(x) for x in tile_grid_size),
    )
    return clahe.apply(img)


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------
def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Unsharp-mask style sharpening.

    Formula:
        sharp = original + strength * (original - blurred)

    Uses a 3x3 Gaussian blur (sigma=1.0) as the blur operator.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    strength : float
        Sharpening strength. 0 = no-op, 1 = standard sharpening,
        2+ = very aggressive.

    Returns
    -------
    np.ndarray
        uint8 image with same shape.
    """
    img = ensure_uint8(ensure_grayscale(image))
    if strength < 0:
        raise ValueError(f"strength must be >= 0, got {strength}.")

    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
    arr = img.astype(np.float64)
    blur_arr = blurred.astype(np.float64)
    out = arr + float(strength) * (arr - blur_arr)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Histogram helper (shared across pages)
# ---------------------------------------------------------------------------
def compute_histogram(image: np.ndarray, bins: int = 256) -> dict:
    """Return a dict with bins and counts for plotting.

    For bins=256, returns exact counts per intensity value.
    """
    img = ensure_uint8(ensure_grayscale(image))
    counts, edges = np.histogram(img.ravel(), bins=bins, range=(0, 256))
    return {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "bins": int(bins),
    }