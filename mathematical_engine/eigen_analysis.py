"""Eigenvalue/eigenvector validation and inspection helpers.

The eigenvalue residual r = || C v - lambda v || is a numerical proof
that a computed eigenpair is correct. We expose it here so the UI can
display it as mathematical validation.

Public API
----------
- eigenvalue_residual(C, v, lambda_val)
- validate_eigenpairs(C, V, lambdas, tol)
- orthonormality_check(V)
- covariance_matrix(X)
- sort_eigenpairs(eigenvalues, eigenvectors)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------
def eigenvalue_residual(
    C: np.ndarray,
    v: np.ndarray,
    lambda_val: float,
) -> float:
    """||C v - lambda v||_2 for a single eigenpair."""
    Cv = C @ v
    lv = lambda_val * v
    return float(np.linalg.norm(Cv - lv))


def validate_eigenpairs(
    C: np.ndarray,
    V: np.ndarray,
    lambdas: np.ndarray,
    tol: float = 1e-6,
) -> dict:
    """Validate a batch of eigenpairs.

    Parameters
    ----------
    C : (D, D)
    V : (D, k) — columns are eigenvectors
    lambdas : (k,) — matching eigenvalues
    tol : float
        Threshold for "all residuals below this = valid".

    Returns
    -------
    dict with residuals, max, all_below_tolerance.
    """
    if C.ndim != 2 or V.ndim != 2:
        raise ValueError("C and V must be 2D.")
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square, got {C.shape}.")
    if V.shape[0] != C.shape[0]:
        raise ValueError(
            f"V rows ({V.shape[0]}) must match C size ({C.shape[0]})."
        )
    if lambdas.ndim != 1 or lambdas.size != V.shape[1]:
        raise ValueError(
            f"lambdas size ({lambdas.size}) must match V columns "
            f"({V.shape[1]})."
        )

    residuals = [
        eigenvalue_residual(C, V[:, i], float(lambdas[i]))
        for i in range(V.shape[1])
    ]
    max_res = max(residuals) if residuals else 0.0

    return {
        "residuals": residuals,
        "max_residual": float(max_res),
        "tolerance": float(tol),
        "all_below_tolerance": bool(max_res <= tol),
    }


# ---------------------------------------------------------------------------
# Orthonormality
# ---------------------------------------------------------------------------
def orthonormality_check(V: np.ndarray, tol: float = 1e-6) -> dict:
    """Check V^T V ≈ I for a matrix of column vectors.

    For orthonormal columns, V^T V should be the identity.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got {V.shape}.")
    G = V.T @ V
    k = V.shape[1]
    identity = np.eye(k)
    max_dev = float(np.max(np.abs(G - identity)))
    return {
        "gram_matrix_diag": G.diagonal().tolist(),
        "max_off_identity": max_dev,
        "is_orthonormal": bool(max_dev <= tol),
    }


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Sample covariance of X (N, D): C = Z^T Z / (N-1)."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}.")
    N = X.shape[0]
    if N < 2:
        raise ValueError(f"Need >= 2 samples, got {N}.")
    Z = X - X.mean(axis=0)
    return (Z.T @ Z) / (N - 1)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------
def sort_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort eigenpairs by eigenvalue descending."""
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be 1D.")
    if eigenvectors.ndim != 2:
        raise ValueError("eigenvectors must be 2D.")
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]