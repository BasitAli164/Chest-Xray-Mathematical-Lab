"""SVD — Singular Value Decomposition and low-rank image compression.

Core identity:
    A = U Σ V^T
    A_k = U_k Σ_k V_k^T     (rank-k approximation)

Public API
----------
- svd_decompose(A)
- low_rank_approximate(A, k)
- reconstruction_error(A, Ak)
- compression_ratio(original_shape, k)
- singular_value_summary(singular_values)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SVDResult:
    U: np.ndarray            # (m, min(m,n))
    S: np.ndarray            # (min(m,n),)
    Vt: np.ndarray           # (min(m,n), n)
    k_used: int
    reconstruction: np.ndarray
    relative_error: float
    absolute_error_frobenius: float
    compression_ratio: float


# ---------------------------------------------------------------------------
# Full SVD
# ---------------------------------------------------------------------------
def svd_decompose(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (U, S, Vt) from NumPy's LAPACK-backed SVD.

    Parameters
    ----------
    A : np.ndarray (m, n)
        Input matrix (2D). Typically a single grayscale image.

    Returns
    -------
    U  : (m, min(m,n))
    S  : (min(m,n),)  sorted descending
    Vt : (min(m,n), n)
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError(f"Expected 2D array, got {getattr(A, 'shape', None)}.")
    U, S, Vt = np.linalg.svd(A.astype(np.float64), full_matrices=False)
    return U, S, Vt


# ---------------------------------------------------------------------------
# Low-rank approximation
# ---------------------------------------------------------------------------
def low_rank_approximate(A: np.ndarray, k: int) -> SVDResult:
    """Reconstruct A using only the top-k singular values/vectors.

    A_k = U_k Σ_k V_k^T  (sum of k rank-1 outer products)

    Parameters
    ----------
    A : np.ndarray (m, n)
    k : int
        Number of singular values to keep. 1 <= k <= min(m, n).

    Returns
    -------
    SVDResult
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError(f"Expected 2D array, got {getattr(A, 'shape', None)}.")

    m, n = A.shape
    kmax = min(m, n)
    if not (1 <= k <= kmax):
        raise ValueError(
            f"k must be in [1, {kmax}], got {k}."
        )

    U, S, Vt = np.linalg.svd(A.astype(np.float64), full_matrices=False)

    Uk = U[:, :k]
    Sk = S[:k]
    Vtk = Vt[:k, :]

    # Reconstruction: sum of rank-1 outer products (educational form)
    # Vectorized equivalent: Uk @ diag(Sk) @ Vtk
    Ak = (Uk * Sk) @ Vtk

    # Errors
    diff = A.astype(np.float64) - Ak
    abs_err = float(np.linalg.norm(diff, ord="fro"))
    denom = float(np.linalg.norm(A.astype(np.float64), ord="fro"))
    rel_err = abs_err / denom if denom > 0 else 0.0

    # Compression ratio (float storage comparison)
    original_floats = m * n
    stored_floats = k * (m + n + 1)   # Uk (m*k) + Vtk (k*n) + Sk (k)
    ratio = original_floats / stored_floats if stored_floats > 0 else 0.0

    return SVDResult(
        U=U,
        S=S,
        Vt=Vt,
        k_used=k,
        reconstruction=Ak,
        relative_error=rel_err,
        absolute_error_frobenius=abs_err,
        compression_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def reconstruction_error(A: np.ndarray, Ak: np.ndarray) -> dict:
    """Return Frobenius norm, relative error, MSE, and PSNR.

    For image uint8 arrays, PSNR = 20*log10(255 / RMSE).
    """
    if A.shape != Ak.shape:
        raise ValueError(
            f"Shape mismatch: A{A.shape} vs Ak{Ak.shape}."
        )
    Af = A.astype(np.float64)
    Akf = Ak.astype(np.float64)
    diff = Af - Akf
    abs_err = float(np.linalg.norm(diff, ord="fro"))
    denom = float(np.linalg.norm(Af, ord="fro"))
    rel_err = abs_err / denom if denom > 0 else 0.0
    mse = float(np.mean(diff ** 2))
    if mse <= 0:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    return {
        "frobenius_error": abs_err,
        "relative_error": rel_err,
        "mse": mse,
        "psnr_db": psnr,
    }


def compression_ratio(original_shape: tuple[int, int], k: int) -> float:
    """Ratio of floats needed for full vs rank-k storage.

    full   : m * n
    rank-k : k * (m + n + 1)
    """
    m, n = original_shape
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    stored = k * (m + n + 1)
    return (m * n) / stored if stored > 0 else 0.0


# ---------------------------------------------------------------------------
# Singular value summary
# ---------------------------------------------------------------------------
def singular_value_summary(S: np.ndarray, top: int = 10) -> dict:
    """Return total energy, top-k list, and cumulative energy ratios."""
    if not isinstance(S, np.ndarray) or S.ndim != 1:
        raise ValueError(f"Expected 1D singular-value array, got {getattr(S, 'shape', None)}.")
    total = float(np.sum(S ** 2))
    cumulative = np.cumsum(S ** 2) / total if total > 0 else np.zeros_like(S)
    return {
        "n_singular_values": int(S.size),
        "max": float(S.max()) if S.size else 0.0,
        "min": float(S.min()) if S.size else 0.0,
        "sum_squared": total,
        "top_k": [
            {"index": int(i), "value": float(S[i]),
             "cumulative_energy": float(cumulative[i])}
            for i in range(min(top, S.size))
        ],
    }