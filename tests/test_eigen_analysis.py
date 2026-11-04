"""Tests for mathematical_engine.eigen_analysis."""

from __future__ import annotations

import numpy as np
import pytest

from mathematical_engine.eigen_analysis import (
    covariance_matrix,
    eigenvalue_residual,
    orthonormality_check,
    sort_eigenpairs,
    validate_eigenpairs,
)


def test_covariance_matrix_shape():
    X = np.random.default_rng(0).normal(size=(10, 5))
    C = covariance_matrix(X)
    assert C.shape == (5, 5)
    assert np.allclose(C, C.T)      # symmetric


def test_covariance_matrix_diag_variance():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    C = covariance_matrix(X)
    # Variance of column 0 with ddof=1
    assert C[0, 0] == pytest.approx(np.var(X[:, 0], ddof=1))


def test_covariance_matrix_rejects_single_sample():
    with pytest.raises(ValueError):
        covariance_matrix(np.zeros((1, 5)))


def test_eigenvalue_residual_zero_for_true_eigenpair():
    # Any symmetric matrix eigenpairs are exact via eigh
    C = np.array([[2.0, 1.0], [1.0, 2.0]])
    vals, vecs = np.linalg.eigh(C)
    for i in range(2):
        r = eigenvalue_residual(C, vecs[:, i], vals[i])
        assert r < 1e-9


def test_eigenvalue_residual_positive_for_wrong_pair():
    C = np.array([[2.0, 1.0], [1.0, 2.0]])
    v = np.array([1.0, 0.0])  # not an eigenvector
    r = eigenvalue_residual(C, v, 3.0)
    assert r > 0.1


def test_validate_eigenpairs():
    C = np.array([[3.0, 1.0], [1.0, 3.0]])
    vals, vecs = np.linalg.eigh(C)
    report = validate_eigenpairs(C, vecs, vals, tol=1e-6)
    assert report["all_below_tolerance"]


def test_validate_eigenpairs_shape_mismatch():
    C = np.eye(3)
    V = np.eye(2)
    lambdas = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        validate_eigenpairs(C, V, lambdas)


def test_orthonormality_check_true():
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(5, 3)))
    report = orthonormality_check(Q, tol=1e-6)
    assert report["is_orthonormal"]


def test_orthonormality_check_false():
    V = np.array([[1.0, 0.0], [1.0, 0.0]])  # columns not orthonormal
    report = orthonormality_check(V, tol=1e-6)
    assert not report["is_orthonormal"]


def test_sort_eigenpairs():
    vals = np.array([1.0, 5.0, 3.0])
    vecs = np.eye(3)
    v_sorted, m_sorted = sort_eigenpairs(vals, vecs)
    assert np.array_equal(v_sorted, np.array([5.0, 3.0, 1.0]))
    # Corresponding columns should follow
    assert np.array_equal(m_sorted[:, 0], vecs[:, 1])