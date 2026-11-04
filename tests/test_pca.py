"""Tests for core.pca — the mathematical equivalence of standard vs dual PCA
is verified on small examples."""

from __future__ import annotations

import numpy as np
import pytest

from core.pca import (
    choose_n_for_variance,
    explained_variance_ratio,
    matrix_row_to_image,
    pca_dual,
    pca_standard,
    reconstruct,
    stack_to_matrix,
)


def _small_data(N=5, D=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(N, D)).astype(np.float64)


# ---------------------------------------------------------------------------
# Standard vs Dual equivalence
# ---------------------------------------------------------------------------
def test_standard_and_dual_give_same_eigenvalues():
    X = _small_data(N=6, D=4, seed=1)
    k = 3
    s = pca_standard(X, n_components=k)
    d = pca_dual(X, n_components=k)
    assert np.allclose(s.eigenvalues, d.eigenvalues, atol=1e-6)


def test_standard_and_dual_give_same_evr():
    X = _small_data(N=6, D=4, seed=2)
    k = 3
    s = pca_standard(X, n_components=k)
    d = pca_dual(X, n_components=k)
    assert np.allclose(s.explained_variance_ratio,
                       d.explained_variance_ratio, atol=1e-6)


def test_standard_and_dual_give_same_reconstruction():
    X = _small_data(N=6, D=4, seed=3)
    k = 3
    s = pca_standard(X, n_components=k)
    d = pca_dual(X, n_components=k)
    # Reconstructions should be numerically identical (up to sign of PCs)
    assert np.allclose(s.reconstruction, d.reconstruction, atol=1e-4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_pca_rejects_1d():
    with pytest.raises(ValueError):
        pca_dual(np.zeros(5), n_components=1)


def test_pca_rejects_too_few_samples():
    with pytest.raises(ValueError):
        pca_dual(np.zeros((1, 4)), n_components=1)


def test_pca_rejects_bad_k():
    X = np.zeros((5, 4))
    with pytest.raises(ValueError):
        pca_dual(X, n_components=10)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def test_evr_sums_to_one():
    eigenvalues = np.array([4.0, 3.0, 2.0, 1.0])
    evr = explained_variance_ratio(eigenvalues)
    assert evr.sum() == pytest.approx(1.0)


def test_evr_zero_total():
    evr = explained_variance_ratio(np.zeros(5))
    assert np.all(evr == 0)


def test_choose_n_for_variance():
    cum = np.array([0.5, 0.8, 0.95, 0.99])
    assert choose_n_for_variance(cum, 0.95) == 3
    assert choose_n_for_variance(cum, 0.99) == 4
    assert choose_n_for_variance(cum, 0.5) == 1


def test_choose_n_not_reached():
    cum = np.array([0.5, 0.8])
    assert choose_n_for_variance(cum, 0.95) == -1


def test_stack_to_matrix_shape():
    stack = np.zeros((5, 4, 4), dtype=np.uint8)
    X = stack_to_matrix(stack)
    assert X.shape == (5, 16)
    assert X.dtype == np.float64


def test_matrix_row_to_image():
    row = np.arange(16, dtype=np.float64)
    img = matrix_row_to_image(row, (4, 4))
    assert img.shape == (4, 4)
    assert img.dtype == np.uint8


def test_reconstruct_identity_components():
    """If components are identity (no compression), reconstruction = X."""
    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    mean = X.mean(axis=0)
    Z = X - mean
    # Identity components: pick the 4 basis vectors
    components = np.eye(4)
    coefficients = Z @ components.T
    X_hat = reconstruct(mean, components, coefficients)
    assert np.allclose(X_hat, X)


# ---------------------------------------------------------------------------
# Realistic small image stack
# ---------------------------------------------------------------------------
def test_pca_on_image_stack():
    rng = np.random.default_rng(7)
    stack = rng.integers(0, 255, size=(10, 8, 8), dtype=np.uint8)
    X = stack_to_matrix(stack)
    k = 5
    res = pca_dual(X, n_components=k)
    assert res.components.shape == (k, 64)
    assert res.reconstruction.shape == (10, 64)
    assert res.cumulative_variance_ratio[-1] <= 1.0 + 1e-9
    assert res.cumulative_variance_ratio[0] > 0