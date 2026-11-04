"""Image loading, validation and grayscale conversion.

All validation logic lives here so the Streamlit UI stays thin and every
error path is testable with pytest.

Public API
----------
- ALLOWED_EXTENSIONS / MAX_FILE_SIZE_MB / MAX_DIMENSION / MIN_DIMENSION
- ImageMetadata            (dataclass)
- validate_extension()     -> str
- validate_size()          -> None (raises)
- load_image()             -> np.ndarray
- to_grayscale()           -> np.ndarray (2D uint8)
- build_metadata()         -> ImageMetadata
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: tuple[str, ...] = ("png", "jpg", "jpeg")
MAX_FILE_SIZE_MB: int = 15
MAX_DIMENSION: int = 4096
MIN_DIMENSION: int = 32


# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageMetadata:
    """Structured metadata derived from an uploaded image.

    All numeric fields are computed from the *actual* grayscale array,
    never estimated or hardcoded.
    """

    width: int
    height: int
    original_mode: str          # e.g. "RGB", "L", "RGBA"
    original_dtype: str         # e.g. "uint8", "int32"
    array_shape: tuple[int, int]
    array_dtype: str
    intensity_min: int
    intensity_max: int
    intensity_mean: float
    intensity_median: float
    intensity_std: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_extension(filename: str) -> str:
    """Return lowercase extension without dot, or raise ValueError.

    Parameters
    ----------
    filename : str
        Original uploaded file name, e.g. "chest_xray.PNG".

    Returns
    -------
    str
        One of "png", "jpg", "jpeg".

    Raises
    ------
    ValueError
        If filename has no extension or extension is unsupported.
    """
    if not filename or "." not in filename:
        raise ValueError(
            "Filename has no extension. Please upload a PNG, JPG or JPEG image."
        )

    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '.{ext}'. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}."
        )

    return ext


def validate_size(file_bytes: bytes) -> None:
    """Raise ValueError if the raw bytes exceed MAX_FILE_SIZE_MB.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content from the Streamlit uploader.

    Raises
    ------
    ValueError
        If the payload is empty or larger than the configured limit.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum allowed is {MAX_FILE_SIZE_MB} MB."
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Decode uploaded bytes into a NumPy array.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Used to validate the extension.

    Returns
    -------
    np.ndarray
        - (H, W, 3) uint8 for RGB
        - (H, W, 4) uint8 for RGBA
        - (H, W)   uint8 for already-grayscale input

    Raises
    ------
    ValueError
        If extension, size, decodability or dimensions are invalid.
    """
    validate_extension(filename)
    validate_size(file_bytes)

    try:
        pil_img = Image.open(BytesIO(file_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError(
            "File could not be decoded as an image. "
            "It may be corrupted or not actually a PNG/JPG/JPEG."
        ) from exc

    # Cheap dimension check before converting to numpy.
    width, height = pil_img.size

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ValueError(
            f"Image too small ({width}x{height}px). "
            f"Minimum dimension is {MIN_DIMENSION}px."
        )

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ValueError(
            f"Image too large ({width}x{height}px). "
            f"Maximum dimension is {MAX_DIMENSION}px for memory safety."
        )

    arr = np.array(pil_img)

    if arr.ndim not in (2, 3):
        raise ValueError(
            f"Unexpected image array shape {arr.shape}. "
            "Expected 2D (grayscale) or 3D (color)."
        )

    return arr


# ---------------------------------------------------------------------------
# Grayscale conversion
# ---------------------------------------------------------------------------
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert any supported image to a 2D uint8 grayscale array.

    Grayscale formula (ITU-R BT.601 luma, same as OpenCV's BGR2GRAY):
        Y = 0.299*R + 0.587*G + 0.114*B

    - Already-2D input   : returned as a contiguous uint8 copy
    - 16-bit / float     : min-max normalized to [0,255] uint8
    - RGBA / gray+alpha  : alpha channel dropped
    - 3+ channels        : first three channels used, then luma formula

    Parameters
    ----------
    image : np.ndarray
        Input array of shape (H, W) or (H, W, C).

    Returns
    -------
    np.ndarray
        Contiguous (H, W) uint8 array.
    """
    # ---- Case 1: already 2D ----
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image.copy())

        # Non-uint8 (e.g. 16-bit PNG, float) -> scale to 8-bit
        arr = image.astype(np.float64)
        arr = arr - arr.min()
        rng = arr.max()

        if rng > 0:
            arr = arr * (255.0 / rng)

        return np.ascontiguousarray(arr.astype(np.uint8))

    # ---- Case 2: 3D ----
    if image.ndim == 3:
        channels = image.shape[2]

        if channels == 1:                      # (H, W, 1)
            return np.ascontiguousarray(image[:, :, 0].astype(np.uint8))

        if channels == 2:                      # grayscale + alpha
            return np.ascontiguousarray(image[:, :, 0].astype(np.uint8))

        if channels >= 3:                      # RGB or RGBA
            rgb = image[:, :, :3].astype(np.float64)
            gray = (
                0.299 * rgb[:, :, 0]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            )
            return np.ascontiguousarray(
                np.clip(np.round(gray), 0, 255).astype(np.uint8)
            )

    raise ValueError(
        f"Cannot convert image of shape {image.shape} to grayscale."
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def build_metadata(
    original: np.ndarray,
    gray: np.ndarray,
    original_mode: Optional[str] = None,
) -> ImageMetadata:
    """Compute ImageMetadata from the actual arrays.

    Parameters
    ----------
    original : np.ndarray
        Original decoded image (any shape).
    gray : np.ndarray
        Grayscale (H, W) uint8 array.
    original_mode : str, optional
        PIL mode string if known ("RGB", "L", "RGBA"). If None, a
        synthetic label is generated from the array shape.

    Returns
    -------
    ImageMetadata
    """
    h, w = gray.shape

    if original_mode is None:
        if original.ndim == 2:
            original_mode = "L (grayscale)"
        elif original.ndim == 3:
            original_mode = f"C{original.shape[2]} (color)"

    return ImageMetadata(
        width=w,
        height=h,
        original_mode=original_mode,
        original_dtype=str(original.dtype),
        array_shape=(h, w),
        array_dtype=str(gray.dtype),
        intensity_min=int(gray.min()),
        intensity_max=int(gray.max()),
        intensity_mean=float(gray.mean()),
        intensity_median=float(np.median(gray)),
        intensity_std=float(gray.std()),
    )