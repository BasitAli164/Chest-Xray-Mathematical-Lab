"""Mathematical feature extraction from grayscale images.

Features are **mathematical characteristics** — NOT clinical indicators.
The app must never present these as disease signals.

Public API
----------
- intensity_statistics(image)
- gradient_statistics(image)
- edge_density_statistics(image)
- histogram_statistics(image)
- glcm_features(image, distances, angles, levels)
- extract_all_features(image)
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from core.edge_detection import (
    canny,
    edge_density,
    gradient_magnitude,
    sobel_x,
    sobel_y,
)
from core.image_preprocessing import ensure_grayscale, ensure_uint8


# ---------------------------------------------------------------------------
# Intensity statistics
# ---------------------------------------------------------------------------
def intensity_statistics(image: np.ndarray) -> dict:
    """Global intensity statistics of a 2D uint8 image.

    Returns
    -------
    dict with:
        min, max, mean, median, std, variance, sum, size, dtype, shape,
        p10, p25, p75, p90   (percentiles)
        skewness, kurtosis   (dimensionless shape statistics)
    """
    img = ensure_uint8(ensure_grayscale(image))
    flat = img.ravel().astype(np.float64)
    m = float(flat.mean())
    s = float(flat.std())
    # Skewness: E[((x - mu)/sigma)^3]
    if s > 0:
        skew = float(np.mean(((flat - m) / s) ** 3))
    else:
        skew = 0.0
    # Excess kurtosis: E[((x - mu)/sigma)^4] - 3
    if s > 0:
        kurt = float(np.mean(((flat - m) / s) ** 4)) - 3.0
    else:
        kurt = 0.0

    p10, p25, p75, p90 = np.percentile(flat, [10, 25, 75, 90])
    return {
        "min": int(flat.min()),
        "max": int(flat.max()),
        "mean": m,
        "median": float(np.median(flat)),
        "std": s,
        "variance": s * s,
        "sum": float(flat.sum()),
        "size": int(flat.size),
        "dtype": str(img.dtype),
        "shape": list(img.shape),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "skewness": skew,
        "kurtosis": kurt,
    }


# ---------------------------------------------------------------------------
# Gradient statistics
# ---------------------------------------------------------------------------
def gradient_statistics(image: np.ndarray) -> dict:
    """Statistics over the gradient-magnitude map."""
    img = ensure_uint8(ensure_grayscale(image))
    gx = sobel_x(img)
    gy = sobel_y(img)
    mag = gradient_magnitude(gx, gy)

    return {
        "gx_mean": float(gx.mean()),
        "gx_std": float(gx.std()),
        "gy_mean": float(gy.mean()),
        "gy_std": float(gy.std()),
        "magnitude_mean": float(mag.mean()),
        "magnitude_std": float(mag.std()),
        "magnitude_max": float(mag.max()),
        "magnitude_p95": float(np.percentile(mag, 95)),
    }


# ---------------------------------------------------------------------------
# Edge density statistics
# ---------------------------------------------------------------------------
def edge_density_statistics(image: np.ndarray) -> dict:
    """Edge density computed via Canny and Sobel-magnitude thresholding."""
    img = ensure_uint8(ensure_grayscale(image))

    # Canny-based
    canny_map = canny(img, low_threshold=50.0, high_threshold=150.0)
    density_canny = edge_density(canny_map, threshold=0)

    # Sobel-threshold based
    gx = sobel_x(img)
    gy = sobel_y(img)
    mag = gradient_magnitude(gx, gy)
    threshold = float(mag.mean() + mag.std())
    density_sobel = float(np.mean(mag > threshold))

    return {
        "edge_density_canny": density_canny,
        "edge_density_sobel_threshold": density_sobel,
        "sobel_threshold_value": threshold,
        "canny_edge_pixels": int(np.count_nonzero(canny_map)),
        "total_pixels": int(img.size),
    }


# ---------------------------------------------------------------------------
# Histogram statistics
# ---------------------------------------------------------------------------
def histogram_statistics(image: np.ndarray) -> dict:
    """Histogram shape descriptors: entropy, uniformity, peak intensity."""
    img = ensure_uint8(ensure_grayscale(image))
    flat = img.ravel()

    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    hist_norm = hist.astype(np.float64) / max(flat.size, 1)

    # Shannon entropy
    with np.errstate(divide="ignore", invalid="ignore"):
        p = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(p * np.log2(p)))

    # Uniformity: sum(p^2)
    uniformity = float(np.sum(hist_norm ** 2))

    # Peak intensity (mode)
    peak = int(np.argmax(hist))

    return {
        "entropy_bits": entropy,
        "uniformity": uniformity,
        "peak_intensity": peak,
        "peak_count": int(hist[peak]),
        "nonzero_bins": int(np.count_nonzero(hist)),
    }


# ---------------------------------------------------------------------------
# GLCM texture (optional, from skimage)
# ---------------------------------------------------------------------------
def glcm_features(
    image: np.ndarray,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 8,
) -> dict:
    """Gray-Level Co-occurrence Matrix texture features.

    NOTE: This uses scikit-image's optimized `graycomatrix`. The individual
    GLCM cell updates are not exposed for step-by-step tracing — only the
    final texture properties are available.

    Parameters
    ----------
    image : np.ndarray
        2D uint8 grayscale image.
    distances : tuple of int
        Pixel-distance offsets for the co-occurrence matrix.
    angles : tuple of float
        Angles (radians) at which GLCM is computed.
    levels : int
        Number of quantization levels (image is rescaled to [0, levels-1]).

    Returns
    -------
    dict with mean/std of contrast, dissimilarity, homogeneity, energy,
    correlation across all (distance, angle) pairs.
    """
    img = ensure_uint8(ensure_grayscale(image))
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}.")

    # Quantize to [0, levels-1]
    quantized = (img.astype(np.float64) / 256.0 * levels).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)

    glcm = graycomatrix(
        quantized,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    out: dict = {}
    for prop in ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation"]:
        arr = graycoprops(glcm, prop)      # (n_distances, n_angles)
        out[f"glcm_{prop}_mean"] = float(arr.mean())
        out[f"glcm_{prop}_std"] = float(arr.std())
    return out


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def extract_all_features(
    image: np.ndarray,
    include_glcm: bool = False,
) -> dict:
    """Run every feature extractor and merge into a single dict.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale uint8 image.
    include_glcm : bool
        GLCM is slightly slower; enable only if needed.

    Returns
    -------
    dict with sub-dicts `intensity`, `gradient`, `edges`, `histogram`,
    and optionally `glcm`.
    """
    img = ensure_uint8(ensure_grayscale(image))
    result = {
        "intensity": intensity_statistics(img),
        "gradient": gradient_statistics(img),
        "edges": edge_density_statistics(img),
        "histogram": histogram_statistics(img),
    }
    if include_glcm:
        result["glcm"] = glcm_features(img)
    return result