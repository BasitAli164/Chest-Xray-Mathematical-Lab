"""Pixel-level convolution tracing.

Produces an exact, verifiable step-by-step trace for a single output
pixel of a convolution operation. Every number reported comes directly
from the arrays — nothing is fabricated.

Public API
----------
- trace_convolution_pixel(image, kernel, row, col, padding="reflect")
- trace_sobel_pixel(image, row, col, axis="x")
"""

from __future__ import annotations

import numpy as np

from core.convolution import pad_image, validate_kernel
from core.edge_detection import SOBEL_X, SOBEL_Y


# ---------------------------------------------------------------------------
# Core trace
# ---------------------------------------------------------------------------
def trace_convolution_pixel(
    image: np.ndarray,
    kernel: np.ndarray,
    row: int,
    col: int,
    padding: str = "reflect",
) -> dict:
    """Trace the exact convolution arithmetic at pixel (row, col).

    Returns
    -------
    dict
        {
            "pixel": [row, col],
            "kernel_shape": [kh, kw],
            "kernel_values": [[...]],
            "padding": padding,
            "neighborhood": [[...]],           # actual local patch
            "multiplications": [              # element-wise (I * K)
                {"i": ..., "j": ..., "image": ..., "kernel": ...,
                 "product": ...},
                ...
            ],
            "sum": float,
            "result": float,
        }

    Raises
    ------
    ValueError
        If image is not 2D, kernel invalid, or pixel out of range.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}.")
    validate_kernel(kernel)

    H, W = image.shape
    if not (0 <= row < H and 0 <= col < W):
        raise IndexError(
            f"Pixel ({row}, {col}) out of range for image {image.shape}."
        )

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = pad_image(image.astype(np.float64), pad_h, pad_w, mode=padding)

    # The (kh x kw) window for output pixel (row, col) lives at
    # padded[row : row+kh, col : col+kw].
    window = padded[row:row + kh, col:col + kw]

    multiplications: list[dict] = []
    acc = 0.0
    K = kernel.astype(np.float64)
    for u in range(kh):
        for v in range(kw):
            iv = float(window[u, v])
            kv = float(K[u, v])
            p = iv * kv
            acc += p
            multiplications.append(
                {
                    "u": u,
                    "v": v,
                    "image_value": iv,
                    "kernel_value": kv,
                    "product": p,
                }
            )

    return {
        "pixel": [int(row), int(col)],
        "kernel_shape": [int(kh), int(kw)],
        "kernel_values": K.tolist(),
        "padding": padding,
        "neighborhood": window.tolist(),
        "multiplications": multiplications,
        "sum": float(acc),
        "result": float(acc),
    }


# ---------------------------------------------------------------------------
# Sobel convenience wrappers
# ---------------------------------------------------------------------------
def trace_sobel_pixel(
    image: np.ndarray,
    row: int,
    col: int,
    axis: str = "x",
) -> dict:
    """Trace Sobel X or Y at one pixel."""
    if axis == "x":
        kernel = SOBEL_X
    elif axis == "y":
        kernel = SOBEL_Y
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
    out = trace_convolution_pixel(image, kernel, row, col, padding="reflect")
    out["axis"] = axis
    return out