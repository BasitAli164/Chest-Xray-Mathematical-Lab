"""Tests for core.svd."""

from __future__ import annotations

import numpy as np
import pytest

from core.svd import (
    compression_ratio,
    low_rank_approximate,
    reconstruction_error,
    singular_value_summary,
    svd_decompose,
)


def _sample_matrix(m=20, n=15, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 50, size=(m, n)).astype(np.float64) + 128


# ---------------------------------------------------------------------------
# SVD decomposition
# ---------------------------------------------------------------------------
def test_svd_shapes():
    A = _sample_matrix(10, 6)
    U, S, Vt = svd_decompose(A)
    assert U.shape == (10, 6)
    assert S.shape == (6,)
    assert Vt.shape == (6, 6)


def test_svd_singular_values_sorted():
    A = _sample_matrix()
    _, S, _ = svd_decompose(A)
    assert np.all(np.diff(S) <= 0)  # descending


def test_svd_reconstruction_identity():
    A = _sample_matrix()
    U, S, Vt = svd_decompose(A)
    A_rec = U @ np.diag(S) @ Vt
    assert np.allclose(A_rec, A, atol=1e-6)


def test_svd_orthogonality():
    A = _sample_matrix(8, 5)
    U, S, Vt = svd_decompose(A)
    # U^T U = I, V^T V = I (economy)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-9)
    assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-9)


def test_svd_rejects_1d():
    with pytest.raises(ValueError):
        svd_decompose(np.zeros(5))


# ---------------------------------------------------------------------------
# Low-rank approximation
# ---------------------------------------------------------------------------
def test_low_rank_k_equals_1_returns_rank1():
    A = _sample_matrix(10, 8)
    res = low_rank_approximate(A, k=1)
    assert res.reconstruction.shape == A.shape
    # Rank-1 matrix: all rows proportional
    assert np.linalg.matrix_rank(res.reconstruction, tol=1e-6) == 1


def test_low_rank_k_equals_rank_returns_original():
    A = _sample_matrix(6, 4)
    k_full = min(A.shape)
    res = low_rank_approximate(A, k=k_full)
    assert np.allclose(res.reconstruction, A, atol=1e-6)
    assert res.relative_error < 1e-9


def test_low_rank_error_decreases_with_k():
    A = _sample_matrix(15, 10)
    res1 = low_rank_approximate(A, k=1)
    res3 = low_rank_approximate(A, k=3)
    res5 = low_rank_approximate(A, k=5)
    assert res1.relative_error > res3.relative_error > res5.relative_error


def test_low_rank_invalid_k():
    A = _sample_matrix(4, 4)
    with pytest.raises(ValueError):
        low_rank_approximate(A, k=0)
    with pytest.raises(ValueError):
        low_rank_approximate(A, k=10)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_reconstruction_error_zero():
    A = _sample_matrix(5, 5)
    errs = reconstruction_error(A, A)
    assert errs["relative_error"] < 1e-9
    assert errs["mse"] < 1e-6
    assert np.isinf(errs["psnr_db"]) or errs["psnr_db"] > 100


def test_reconstruction_error_positive():
    A = _sample_matrix(5, 5)
    Ak = A + 5.0
    errs = reconstruction_error(A, Ak)
    assert errs["relative_error"] > 0
    assert errs["mse"] > 0


def test_compression_ratio():
    # m=100, n=100, k=10: 10000 / (10*201) = ~4.975x
    ratio = compression_ratio((100, 100), k=10)
    assert ratio == pytest.approx(10000 / (10 * 201), rel=1e-6)


def test_singular_value_summary():
    S = np.array([10.0, 5.0, 2.0, 1.0])
    summary = singular_value_summary(S, top=3)
    assert summary["n_singular_values"] == 4
    assert summary["max"] == 10.0
    # Cumulative energy at index 2: (100+25+4) / (100+25+4+1)
    expected_cum = (100 + 25 + 4) / (100 + 25 + 4 + 1)
    assert summary["top_k"][2]["cumulative_energy"] == pytest.approx(
        expected_cum, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Eckart-Young theorem numerical verification
# ---------------------------------------------------------------------------
def test_eckart_young_optimal():
    """||A - A_k||_F^2 = sum_{i>k} sigma_i^2 — verify with our implementation."""
    A = _sample_matrix(15, 10, seed=99)
    _, S, _ = svd_decompose(A)
    k = 4
    res = low_rank_approximate(A, k=k)
    # Theoretical Frobenius error squared = sum of tail singular values squared
    expected_sq = float(np.sum(S[k:] ** 2))
    actual_sq = float(res.absolute_error_frobenius ** 2)
    assert actual_sq == pytest.approx(expected_sq, rel=1e-6)