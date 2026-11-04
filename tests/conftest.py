"""Shared pytest fixtures for the Chest X-ray Mathematical Lab.

Fixtures
--------
sample_gray : 64x64 uint8 horizontal gradient (0..255)
sample_rgb  : 32x32x3 uint8 image with known per-channel values
sample_rgba : 32x32x4 uint8 image (alpha dropped by to_grayscale)
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_gray() -> np.ndarray:
    """A 64x64 uint8 horizontal gradient — deterministic and small."""
    ramp = np.linspace(0, 255, 64).astype(np.uint8)
    return np.tile(ramp, (64, 1))


@pytest.fixture
def sample_rgb() -> np.ndarray:
    """A 32x32x3 uint8 image with known channel values.

    - Red channel   : 255 everywhere
    - Green channel : 255 on the top half
    - Blue channel  : 255 on the left half
    """
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    img[0:16, :, 1] = 255
    img[:, 0:16, 2] = 255
    return img


@pytest.fixture
def sample_rgba() -> np.ndarray:
    """A 32x32x4 uint8 image — alpha channel is set to 128."""
    img = np.zeros((32, 32, 4), dtype=np.uint8)
    img[:, :, 0] = 100
    img[:, :, 1] = 150
    img[:, :, 2] = 200
    img[:, :, 3] = 128
    return img