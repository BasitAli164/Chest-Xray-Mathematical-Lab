"""Image-quality metrics with explicit honest disclaimers.

Metrics computed here are **mathematical similarity/difference measures**
between two images. They are NOT proxies for diagnostic quality. Both
images must be the same shape and intensity convention.

Public API
----------
- mse(original, processed)
- psnr(original, processed, data_range=255)
- ssim(original, processed, data_range=255)
- evaluate_all(original, processed)
- DISCLAIMER  (module-level string)
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


DISCLAIMER = (
    "MSE, PSNR and SSIM quantify mathematical difference / similarity "
    "between two images. They are NOT evidence of diagnostic improvement "
    "or clinical validity. Interpret only as image-reconstruction or "
    "transformation fidelity measures, and only when a valid reference "
    "image exists."
)


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------
def mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Mean Squared Error.

    MSE = (1/N) * sum( (A_i - B_i)^2 )

    Parameters
    ----------
    original, processed : np.ndarray
        Same-shape arrays.

    Returns
    -------
    float
    """
    if original.shape != processed.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {processed.shape}."
        )
    a = original.astype(np.float64)
    b = processed.astype(np.float64)
    return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def psnr(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: float = 255.0,
) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    PSNR = 10 * log10( (data_range^2) / MSE )

    Returns +inf when MSE = 0 (identical images).
    """
    m = mse(original, processed)
    if m <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / m))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def ssim(
    original: np.ndarray,
    processed: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 7,
) -> float:
    """Structural Similarity Index (mean over the whole image).

    Parameters
    ----------
    original, processed : np.ndarray
        2D arrays. Must be at least `win_size` in both dimensions.
    data_range : float
        For 8-bit images, 255.
    win_size : int
        Window size; must be odd. Default 7.

    Returns
    -------
    float in [-1, 1], typically [0, 1].
    """
    if original.ndim != 2 or processed.ndim != 2:
        raise ValueError(
            "SSIM requires 2D arrays; got shapes "
            f"{original.shape} and {processed.shape}."
        )
    if original.shape != processed.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {processed.shape}."
        )
    if min(original.shape) < win_size:
        raise ValueError(
            f"Image too small for SSIM: min dim {min(original.shape)} "
            f"< win_size {win_size}."
        )
    # skimage expects float arrays and data_range for the scale
    a = original.astype(np.float64)
    b = processed.astype(np.float64)
    score, _ = structural_similarity(
        a, b, data_range=data_range, win_size=win_size, full=True,
    )
    return float(score)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def evaluate_all(
    original: np.ndarray,
    processed: np.ndarray,
) -> dict:
    """Compute MSE, PSNR, SSIM plus metadata.

    Returns
    -------
    dict:
        "mse"             : float
        "psnr_db"         : float (or inf)
        "ssim"            : float (or None if image too small)
        "shape"           : list
        "disclaimer"      : str
        "notes"           : list[str]  — validity notes
    """
    notes: list[str] = []
    out: dict = {
        "shape": list(original.shape),
        "disclaimer": DISCLAIMER,
    }

    if original.shape != processed.shape:
        notes.append(
            f"Shape mismatch — evaluation skipped "
            f"({original.shape} vs {processed.shape})."
        )
        out["mse"] = None
        out["psnr_db"] = None
        out["ssim"] = None
        out["notes"] = notes
        return out

    out["mse"] = mse(original, processed)
    out["psnr_db"] = psnr(original, processed)

    try:
        if original.ndim == 2 and min(original.shape) >= 7:
            out["ssim"] = ssim(original, processed)
        else:
            out["ssim"] = None
            notes.append(
                "SSIM not computed (image is too small or not 2D)."
            )
    except Exception as exc:
        out["ssim"] = None
        notes.append(f"SSIM failed: {exc}")

    out["notes"] = notes
    return out