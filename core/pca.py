"""PCA — Principal Component Analysis for images.

Two implementations are provided:

- `pca_standard(X)` — naive eigen-decomposition of the (D x D) covariance
  matrix. Educational baseline; only viable when D is small.

- `pca_dual(X)` — memory-efficient dual PCA using the (N x N) Gram matrix.
  Mathematically equivalent to standard PCA when N << D, but uses
  N^2 memory instead of D^2.

Both return the same conceptual results:
    - mean image
    - top-k eigenvectors in pixel space (principal components)
    - explained variance ratio
    - reconstruction at chosen k
    - projection coefficients

Public API
----------
- pca_standard(X, n_components)
- pca_dual(X, n_components)
- reconstruct(mean, components, coefficients)
- explained_variance_ratio(eigenvalues)
- choose_n_for_variance(cumulative_ratio, target)

Data convention
---------------
X : (N, H*W) float64 — each row is one flattened, mean-centered image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses for structured returns
# ---------------------------------------------------------------------------
@dataclass
class PCAResult:
    mean_image: np.ndarray            # (D,) float64
    components: np.ndarray            # (k, D) float64 — top-k eigenvectors
    eigenvalues: np.ndarray           # (k,) float64 — top-k eigenvalues
    explained_variance_ratio: np.ndarray  # (k,) — eigenvalue_i / sum(all)
    cumulative_variance_ratio: np.ndarray  # (k,)
    coefficients: np.ndarray          # (N, k) — projection of X onto components
    reconstruction: np.ndarray        # (N, D) — X approximated with k components
    method: str                       # "standard" or "dual"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_X(X: np.ndarray) -> None:
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D), got shape {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 samples for PCA, got {X.shape[0]}."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf.")


# ---------------------------------------------------------------------------
# Standard PCA (D x D covariance)
# ---------------------------------------------------------------------------
def pca_standard(X: np.ndarray, n_components: int) -> PCAResult:
    """Classical PCA via covariance eigen-decomposition.

    Steps
    -----
    1. mean = column mean of X.
    2. Center: Z = X - mean.
    3. Covariance: C = Z^T Z / (N - 1)   # (D, D)
    4. Eigen-decompose C (symmetric) -> eigenvalues, eigenvectors.
    5. Sort descending, take top-k.
    6. Project: coefficients = Z @ V_k  # (N, k)
    7. Reconstruct: X_hat = mean + coefficients @ V_k^T

    Memory: O(D^2). Use only when D is small (< ~5000).

    Parameters
    ----------
    X : np.ndarray (N, D)
    n_components : int

    Returns
    -------
    PCAResult
    """
    _validate_X(X)
    N, D = X.shape
    if not (1 <= n_components <= min(N - 1, D)):
        raise ValueError(
            f"n_components must be in [1, min(N-1, D)] = [1, "
            f"{min(N - 1, D)}], got {n_components}."
        )

    mean = X.mean(axis=0)
    Z = X - mean

    # Covariance matrix (D, D)
    C = (Z.T @ Z) / (N - 1)

    # Symmetric eigen-decomposition (eigenvalues real, eigenvectors orthonormal)
    eigenvalues_all, eigenvectors_all = np.linalg.eigh(C)

    # eigh returns ascending order; we want descending
    order = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_all = eigenvalues_all[order]
    eigenvectors_all = eigenvectors_all[:, order]

    # Clip tiny negatives (numerical noise)
    eigenvalues_all = np.clip(eigenvalues_all, 0.0, None)

    # Top-k
    eigenvalues = eigenvalues_all[:n_components]
    components = eigenvectors_all[:, :n_components].T   # (k, D)

    # Explained variance
    evr = explained_variance_ratio(eigenvalues_all)[:n_components]
    cum = np.cumsum(evr)

    # Projection and reconstruction
    coefficients = Z @ components.T                     # (N, k)
    reconstruction = mean + coefficients @ components   # (N, D)

    return PCAResult(
        mean_image=mean,
        components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=evr,
        cumulative_variance_ratio=cum,
        coefficients=coefficients,
        reconstruction=reconstruction,
        method="standard",
    )


# ---------------------------------------------------------------------------
# Dual PCA (N x N Gram matrix) — memory efficient
# ---------------------------------------------------------------------------
def pca_dual(X: np.ndarray, n_components: int) -> PCAResult:
    """Dual PCA using the (N x N) Gram matrix.

    Mathematical equivalence with standard PCA
    ------------------------------------------
    Let Z = X - mean (N, D). The covariance is C = Z^T Z / (N-1) (D, D).
    The Gram matrix is G = Z Z^T / (N-1) (N, N).

    Any eigenvector v of C corresponds to a left singular vector u of Z:
        C v = lambda v  =>  Z^T Z v = (N-1) lambda v
        => v = Z^T u / sqrt((N-1) lambda) for the associated u with
           G u = lambda u.

    Concretely:
    - Compute eigenvectors of G: u_i (N-dim)
    - Convert to primal eigenvectors: v_i = Z^T u_i / || Z^T u_i ||
    - Eigenvalues carry over (scaled by (N-1) if using covariance).

    For reconstruction, we can skip explicit v_i by using the identity:
        X_hat = mean + (Z V_k) V_k^T
              = mean + Z (Z^T U_k) (U_k^T Z) / scaling
              = mean + Z U_k U_k^T
    when U_k are orthonormal (unit-norm) eigenvectors of G.

    Memory: O(N^2) instead of O(D^2). Essential for real images.

    Parameters
    ----------
    X : np.ndarray (N, D)
    n_components : int

    Returns
    -------
    PCAResult
    """
    _validate_X(X)
    N, D = X.shape
    if not (1 <= n_components <= min(N - 1, D)):
        raise ValueError(
            f"n_components must be in [1, min(N-1, D)] = [1, "
            f"{min(N - 1, D)}], got {n_components}."
        )

    mean = X.mean(axis=0)
    Z = X - mean

    # Gram matrix (N, N)
    G = (Z @ Z.T) / (N - 1)

    # Symmetric eigen-decomposition
    eigenvalues_all, U_all = np.linalg.eigh(G)
    order = np.argsort(eigenvalues_all)[::-1]
    eigenvalues_all = np.clip(eigenvalues_all[order], 0.0, None)
    U_all = U_all[:, order]                       # (N, N)

    # Top-k
    eigenvalues = eigenvalues_all[:n_components]
    U_k = U_all[:, :n_components]                 # (N, k)
    evr = explained_variance_ratio(eigenvalues_all)[:n_components]
    cum = np.cumsum(evr)

    # Convert to primal eigenvectors v_i = Z^T u_i, then normalize to unit.
    # If ||Z^T u_i|| is zero (degenerate), keep as zeros — those components
    # carry no variance.
    V_k = Z.T @ U_k                                # (D, k)
    norms = np.linalg.norm(V_k, axis=0, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)        # avoid division by zero
    components = (V_k / norms).T                   # (k, D)

    # Coefficients: project Z onto components
    coefficients = Z @ components.T                # (N, k)

    # Reconstruction
    reconstruction = mean + coefficients @ components

    return PCAResult(
        mean_image=mean,
        components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=evr,
        cumulative_variance_ratio=cum,
        coefficients=coefficients,
        reconstruction=reconstruction,
        method="dual",
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """EVR_i = lambda_i / sum(lambda).

    Handles the all-zero case gracefully (returns zeros).
    """
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return np.zeros_like(eigenvalues)
    return eigenvalues / total


def reconstruct(
    mean: np.ndarray,
    components: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Reconstruct: X_hat = mean + coefficients @ components.

    Parameters
    ----------
    mean : (D,)
    components : (k, D)
    coefficients : (N, k)

    Returns
    -------
    (N, D) float64
    """
    return mean + coefficients @ components


def choose_n_for_variance(
    cumulative_ratio: np.ndarray,
    target: float = 0.95,
) -> int:
    """Return the smallest k such that cumulative_ratio[k-1] >= target.

    Returns -1 if even the last value doesn't reach `target`.
    """
    if not (0 < target <= 1.0):
        raise ValueError(f"target must be in (0, 1], got {target}.")
    for k in range(1, len(cumulative_ratio) + 1):
        if cumulative_ratio[k - 1] >= target:
            return k
    return -1


# ---------------------------------------------------------------------------
# Convenience: from 3D image stack
# ---------------------------------------------------------------------------
def stack_to_matrix(stack: np.ndarray) -> np.ndarray:
    """Flatten (N, H, W) uint8 stack into (N, H*W) float64."""
    if stack.ndim != 3:
        raise ValueError(
            f"Expected stack (N, H, W), got shape {stack.shape}."
        )
    N = stack.shape[0]
    return stack.reshape(N, -1).astype(np.float64)


def matrix_row_to_image(row: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reshape a single (H*W,) row into a (H, W) image clipped to uint8."""
    return np.clip(np.round(row), 0, 255).astype(np.uint8).reshape(shape)