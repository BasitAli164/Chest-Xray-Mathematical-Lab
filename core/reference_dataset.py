"""Reference dataset management for PCA experiments.

Design
------
PCA requires MULTIPLE images. We do NOT bundle the NIH ChestX-ray14
dataset (too large, licensing constraints). Instead, this module:

1. Provides a **synthetic_chest_like** generator for reproducible demos.
2. Accepts a **user-supplied folder** of reference images (for real
   experiments) with strict safety checks.

Both paths produce a deterministic, preprocessed image stack ready for
PCA.

Public API
----------
- synthesize_reference_stack(n_images, size, seed)
- load_reference_folder(folder_path, size, max_images)
- preprocess_stack(images, target_size)
- describe_stack(stack)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from core.image_loader import load_image, to_grayscale
from core.image_preprocessing import ensure_grayscale, ensure_uint8


# ---------------------------------------------------------------------------
# Synthetic chest-like images
# ---------------------------------------------------------------------------
def _make_synthetic_chest(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic 'chest-like' image with an elliptical shape.

    This is a deterministic function of rng — NOT a real X-ray. It has:
    - A central bright elliptical 'thorax' region.
    - Darker 'lung' ellipses on the left and right.
    - Additive Gaussian noise.

    The goal is a repeatable visual structure that PCA can meaningfully
    decompose. It is clearly labeled as synthetic in all UI output.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy = height / 2.0
    cx = width / 2.0

    # Outer torso ellipse (bright-ish)
    a = 0.42 * height
    b = 0.40 * width
    outer = ((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2

    base = np.full((height, width), 40.0)
    base[outer <= 1.0] = 200.0

    # Two lung ellipses (dark)
    lung_a = 0.28 * height
    lung_b = 0.14 * width
    for offset in (-0.17 * width, +0.17 * width):
        lung = ((yy - cy) / lung_a) ** 2 + ((xx - (cx + offset)) / lung_b) ** 2
        base[lung <= 1.0] = 30.0

    # Add per-image structured variation (small brightness offset + noise)
    brightness_jitter = float(rng.uniform(-25, 25))
    base = base + brightness_jitter
    noise = rng.normal(0, 12, size=(height, width))
    base = base + noise

    return np.clip(base, 0, 255).astype(np.uint8)


def synthesize_reference_stack(
    n_images: int = 20,
    size: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Generate a reproducible stack of synthetic chest-like images.

    Parameters
    ----------
    n_images : int
        Number of images (default 20).
    size : int
        Side length in pixels; images are square (size × size).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Stack of shape (n_images, size, size), dtype uint8.
    """
    if n_images < 2:
        raise ValueError(
            f"Need at least 2 images for PCA, got {n_images}."
        )
    if size < 16:
        raise ValueError(f"size must be >= 16, got {size}.")

    rng = np.random.default_rng(seed)
    stack = np.empty((n_images, size, size), dtype=np.uint8)
    for i in range(n_images):
        stack[i] = _make_synthetic_chest(size, size, rng)
    return stack


# ---------------------------------------------------------------------------
# User-supplied folder
# ---------------------------------------------------------------------------
def load_reference_folder(
    folder_path: str | Path,
    target_size: int = 128,
    max_images: int = 50,
) -> np.ndarray:
    """Load a folder of PNG/JPG/JPEG images as a preprocessed stack.

    Parameters
    ----------
    folder_path : str | Path
        Path to folder containing images.
    target_size : int
        All images resized to (target_size, target_size).
    max_images : int
        Cap on number of images to load (memory safety on Streamlit Cloud).

    Returns
    -------
    np.ndarray
        Stack of shape (n_loaded, target_size, target_size), uint8.

    Raises
    ------
    ValueError
        If folder doesn't exist, is empty, or contains no valid images.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in exts]
    )
    if not files:
        raise ValueError(f"No PNG/JPG/JPEG images in {folder}.")

    files = files[:max_images]
    imgs: list[np.ndarray] = []

    from core.image_preprocessing import resize_max_dimension
    from PIL import Image

    for p in files:
        try:
            data = p.read_bytes()
            orig = load_image(data, p.name)
            gray = to_grayscale(orig)
            gray = ensure_uint8(ensure_grayscale(gray))
            # Resize to target_size (square)
            pil = Image.fromarray(gray)
            pil = pil.resize((target_size, target_size),
                             resample=Image.BILINEAR)
            imgs.append(np.array(pil, dtype=np.uint8))
        except Exception:
            # Skip unreadable images silently — logging would be noisy
            # in the UI. Caller can inspect count vs files.
            continue

    if len(imgs) < 2:
        raise ValueError(
            f"Need at least 2 readable images for PCA. "
            f"Found {len(imgs)} readable in {len(files)} files."
        )

    return np.stack(imgs, axis=0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_stack(
    images: np.ndarray,
    target_size: Optional[int] = None,
) -> np.ndarray:
    """Ensure stack is (N, size, size) uint8.

    If target_size given, images are resized (bilinear) via PIL.
    """
    if images.ndim != 3:
        raise ValueError(
            f"Expected stack (N, H, W), got shape {images.shape}."
        )
    if target_size is not None:
        from PIL import Image
        out = np.empty(
            (images.shape[0], target_size, target_size), dtype=np.uint8
        )
        for i, img in enumerate(images):
            pil = Image.fromarray(img.astype(np.uint8))
            pil = pil.resize((target_size, target_size),
                             resample=Image.BILINEAR)
            out[i] = np.array(pil, dtype=np.uint8)
        return out
    return images.astype(np.uint8, copy=True)


# ---------------------------------------------------------------------------
# Description helper
# ---------------------------------------------------------------------------
def describe_stack(stack: np.ndarray) -> dict:
    """Return a dict with stack metadata (for UI display)."""
    if stack.ndim != 3:
        raise ValueError(f"Expected stack (N, H, W), got {stack.shape}.")
    return {
        "n_images": int(stack.shape[0]),
        "height": int(stack.shape[1]),
        "width": int(stack.shape[2]),
        "dtype": str(stack.dtype),
        "pixel_count_per_image": int(stack.shape[1] * stack.shape[2]),
        "total_bytes": int(stack.nbytes),
        "min": int(stack.min()),
        "max": int(stack.max()),
        "mean": float(stack.mean()),
        "std": float(stack.std()),
    }