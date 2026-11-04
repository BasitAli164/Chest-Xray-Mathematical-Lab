"""Tests for core.evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from core.evaluation import (
    DISCLAIMER,
    evaluate_all,
    mse,
    psnr,
    ssim,
)


def test_mse_identical():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert mse(A, A) == 0.0


def test_mse_known_value():
    A = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    B = np.array([[3, 3], [3, 3]], dtype=np.uint8)
    assert mse(A, B) == pytest.approx(9.0)


def test_mse_shape_mismatch():
    with pytest.raises(ValueError):
        mse(np.zeros((2, 2)), np.zeros((3, 3)))


def test_psnr_identical_is_inf():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert np.isinf(psnr(A, A))


def test_psnr_known_value():
    A = np.zeros((10, 10), dtype=np.uint8)
    B = A.copy()
    B[0, 0] = 10
    # MSE = 10^2/100 = 1 -> PSNR = 10*log10(255^2/1) = 20*log10(255)
    expected = 20 * np.log10(255.0)
    assert psnr(A, B) == pytest.approx(expected, rel=1e-6)


def test_ssim_identical():
    A = np.random.default_rng(0).integers(0, 256, size=(20, 20),
                                          dtype=np.uint8)
    assert ssim(A, A) == pytest.approx(1.0, abs=1e-9)


def test_ssim_different():
    A = np.zeros((20, 20), dtype=np.uint8)
    B = np.full((20, 20), 200, dtype=np.uint8)
    score = ssim(A, B)
    assert -1.0 <= score <= 1.0
    assert score < 0.9


def test_ssim_too_small():
    A = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        ssim(A, A)


def test_evaluate_all_ok():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    B = A.copy()
    B[0, 0] = 5
    m = evaluate_all(A, B)
    assert m["mse"] is not None
    assert m["psnr_db"] is not None
    assert m["ssim"] is not None
    assert m["shape"] == [10, 10]
    assert m["disclaimer"] == DISCLAIMER
    assert m["notes"] == []


def test_evaluate_all_identical():
    A = np.arange(100, dtype=np.uint8).reshape(10, 10)
    m = evaluate_all(A, A)
    assert m["mse"] == 0.0
    assert np.isinf(m["psnr_db"])
    assert m["ssim"] == pytest.approx(1.0, abs=1e-9)


def test_evaluate_all_shape_mismatch():
    A = np.zeros((5, 5), dtype=np.uint8)
    B = np.zeros((6, 6), dtype=np.uint8)
    m = evaluate_all(A, B)
    assert m["mse"] is None
    assert m["psnr_db"] is None
    assert m["ssim"] is None
    assert any("Shape mismatch" in n for n in m["notes"])


def test_disclaimer_is_present():
    assert isinstance(DISCLAIMER, str)
    assert "not" in DISCLAIMER.lower()