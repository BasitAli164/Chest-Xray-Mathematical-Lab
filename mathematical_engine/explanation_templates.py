"""Human-readable derivation builders.

Each builder returns a `list[dict]` where every dict is one step:
    {
        "step":       int,
        "title":      str,
        "expression": str,           # plain text math
        "latex":      str | None,    # optional LaTeX
        "values":     dict | None,   # actual numbers pulled from the trace
    }

Builders MUST only use values that actually come from the computation.
Never invent numbers.
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# Phase 2 builders (matrix ops, statistics, intensity transforms)
# ===========================================================================
def explain_transpose(A: np.ndarray, At: np.ndarray) -> list[dict]:
    m, n = A.shape
    return [
        {
            "step": 1,
            "title": "Input shape",
            "expression": f"A has shape {m} x {n}",
            "latex": r"A \in \mathbb{R}^{m \times n}",
            "values": {"m": m, "n": n},
        },
        {
            "step": 2,
            "title": "Transpose definition",
            "expression": "A^T[j, i] = A[i, j]  for every i in [0,m), j in [0,n)",
            "latex": r"A^{\top}_{j,i} = A_{i,j}",
            "values": None,
        },
        {
            "step": 3,
            "title": "Output shape",
            "expression": f"A^T has shape {At.shape[0]} x {At.shape[1]}",
            "latex": r"A^{\top} \in \mathbb{R}^{n \times m}",
            "values": {"shape": list(At.shape)},
        },
        {
            "step": 4,
            "title": "Spot check",
            "expression": (
                f"A^T[0, 0] = A[0, 0] = {A[0, 0]}  and  "
                f"A^T[0, 1] = A[1, 0] = {A[1, 0] if m > 1 else 'n/a'}"
            ),
            "latex": None,
            "values": {
                "At_00": int(At[0, 0]),
                "A_00": int(A[0, 0]),
            },
        },
    ]


def explain_add(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Input shapes",
            "expression": f"A{A.shape}, B{B.shape} — must match",
            "latex": None,
            "values": {"A": list(A.shape), "B": list(B.shape)},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "C[i, j] = A[i, j] + B[i, j]",
            "latex": r"C_{i,j} = A_{i,j} + B_{i,j}",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": f"C[0,0] = {A[0, 0]} + {B[0, 0]} = {C[0, 0]}",
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "B_00": int(B[0, 0]),
                "C_00": int(C[0, 0]),
            },
        },
    ]


def explain_subtract(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Input shapes",
            "expression": f"A{A.shape}, B{B.shape}",
            "latex": None,
            "values": {"A": list(A.shape), "B": list(B.shape)},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "C[i, j] = A[i, j] - B[i, j]",
            "latex": r"C_{i,j} = A_{i,j} - B_{i,j}",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": f"C[0,0] = {A[0, 0]} - {B[0, 0]} = {C[0, 0]}",
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "B_00": int(B[0, 0]),
                "C_00": int(C[0, 0]),
            },
        },
    ]


def explain_scalar(A: np.ndarray, k: float, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Scalar",
            "expression": f"k = {k}",
            "latex": None,
            "values": {"k": k},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "C[i, j] = k * A[i, j]",
            "latex": r"C_{i,j} = k \cdot A_{i,j}",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": f"C[0,0] = {k} * {A[0, 0]} = {C[0, 0]}",
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "k": k,
                "C_00": float(C[0, 0]),
            },
        },
    ]


def explain_elementwise(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Input shapes",
            "expression": f"A{A.shape}, B{B.shape}",
            "latex": None,
            "values": {"A": list(A.shape), "B": list(B.shape)},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "C[i, j] = A[i, j] * B[i, j]  (Hadamard product)",
            "latex": r"C_{i,j} = A_{i,j} \odot B_{i,j}",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": f"C[0,0] = {A[0, 0]} * {B[0, 0]} = {C[0, 0]}",
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "B_00": int(B[0, 0]),
                "C_00": float(C[0, 0]),
            },
        },
    ]


def explain_matrix_multiply(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> list[dict]:
    m, n = A.shape
    n2, p = B.shape
    return [
        {
            "step": 1,
            "title": "Input shapes",
            "expression": f"A{A.shape}, B{B.shape}  (inner dims {n} == {n2})",
            "latex": None,
            "values": {"A": list(A.shape), "B": list(B.shape)},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "C[i, j] = sum_k A[i, k] * B[k, j]",
            "latex": r"C_{i,j} = \sum_{k=0}^{n-1} A_{i,k} B_{k,j}",
            "values": {"n": n},
        },
        {
            "step": 3,
            "title": "Output shape",
            "expression": f"C has shape {m} x {p}",
            "latex": r"C \in \mathbb{R}^{m \times p}",
            "values": {"shape": [m, p]},
        },
        {
            "step": 4,
            "title": "Spot check: C[0, 0]",
            "expression": " + ".join(
                f"{A[0, k]}*{B[k, 0]}" for k in range(min(n, 4))
            ) + (" + ..." if n > 4 else ""),
            "latex": None,
            "values": {"C_00": float(C[0, 0])},
        },
    ]


def explain_mean(A: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Definition",
            "expression": "mu = (1/N) * sum(x_i)",
            "latex": r"\mu = \frac{1}{N}\sum_{i=1}^{N} x_i",
            "values": None,
        },
        {
            "step": 2,
            "title": "Compute",
            "expression": (
                f"N = {A.size}, sum = {A.sum():.2f}, "
                f"mu = {A.mean():.4f}"
            ),
            "latex": None,
            "values": {
                "N": int(A.size),
                "sum": float(A.sum()),
                "mean": float(A.mean()),
            },
        },
    ]


def explain_variance(A: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Definition",
            "expression": "sigma^2 = (1/N) * sum((x_i - mu)^2)",
            "latex": r"\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2",
            "values": None,
        },
        {
            "step": 2,
            "title": "Compute",
            "expression": (
                f"mu = {A.mean():.4f}, variance = {A.var():.4f}, "
                f"sigma = {A.std():.4f}"
            ),
            "latex": None,
            "values": {
                "mean": float(A.mean()),
                "variance": float(A.var()),
                "std": float(A.std()),
            },
        },
    ]


def explain_linear(
    A: np.ndarray,
    gain: float,
    offset: float,
    C: np.ndarray,
) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Parameters",
            "expression": f"gain g = {gain}, offset b = {offset}",
            "latex": None,
            "values": {"gain": gain, "offset": offset},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "s = g * r + b, clipped to [0, 255]",
            "latex": r"s = g \cdot r + b",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": (
                f"s(0,0) = {gain} * {int(A[0, 0])} + {offset} "
                f"= {float(gain * A[0, 0] + offset):.3f}  "
                f"-> {float(C[0, 0]):.3f}"
            ),
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "raw": float(gain * A[0, 0] + offset),
                "clipped": float(C[0, 0]),
            },
        },
    ]


def explain_gamma(A: np.ndarray, gamma: float, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Parameter",
            "expression": f"gamma = {gamma}",
            "latex": None,
            "values": {"gamma": gamma},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "s = 255 * (r / 255) ^ gamma",
            "latex": r"s = 255 \left(\frac{r}{255}\right)^{\gamma}",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": (
                f"s(0,0) = 255 * ({int(A[0, 0])}/255) ^ {gamma} "
                f"= {float(C[0, 0]):.3f}"
            ),
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "raw": float(255.0 * ((A[0, 0] / 255.0) ** gamma)),
                "clipped": float(C[0, 0]),
            },
        },
    ]


def explain_negative(A: np.ndarray, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Definition",
            "expression": "s = 255 - r",
            "latex": r"s = 255 - r",
            "values": None,
        },
        {
            "step": 2,
            "title": "Spot check (top-left)",
            "expression": f"s(0,0) = 255 - {int(A[0, 0])} = {int(C[0, 0])}",
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "C_00": int(C[0, 0]),
            },
        },
    ]


# ===========================================================================
# Phase 3 builders (enhancement, convolution, edge detection)
# ===========================================================================
def explain_brightness(A: np.ndarray, offset: int, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Parameter",
            "expression": f"offset b = {offset}",
            "latex": None,
            "values": {"offset": offset},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "s = clip(r + b, 0, 255)",
            "latex": r"s = \mathrm{clip}(r + b,\, 0,\, 255)",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": (
                f"s(0,0) = clip({int(A[0, 0])} + {offset}, 0, 255) "
                f"= {int(C[0, 0])}"
            ),
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "raw": int(A[0, 0]) + offset,
                "clipped": int(C[0, 0]),
            },
        },
    ]


def explain_contrast(A: np.ndarray, gain: float, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Parameter",
            "expression": f"gain g = {gain}",
            "latex": None,
            "values": {"gain": gain},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "s = clip(g * (r - 128) + 128, 0, 255)",
            "latex": r"s = \mathrm{clip}(g\,(r - 128) + 128,\, 0,\, 255)",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": (
                f"s(0,0) = clip({gain} * ({int(A[0, 0])} - 128) + 128, "
                f"0, 255) = {int(C[0, 0])}"
            ),
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "raw": float(gain * (int(A[0, 0]) - 128) + 128),
                "clipped": int(C[0, 0]),
            },
        },
    ]


def explain_min_max(A: np.ndarray, C: np.ndarray) -> list[dict]:
    r_min = int(A.min())
    r_max = int(A.max())
    return [
        {
            "step": 1,
            "title": "Range",
            "expression": f"r_min = {r_min}, r_max = {r_max}",
            "latex": None,
            "values": {"r_min": r_min, "r_max": r_max},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "s = (r - r_min) / (r_max - r_min) * 255",
            "latex": r"s = \frac{r - r_{min}}{r_{max} - r_{min}}\cdot 255",
            "values": None,
        },
        {
            "step": 3,
            "title": "Spot check (top-left)",
            "expression": (
                f"s(0,0) = ({int(A[0, 0])} - {r_min}) / "
                f"({r_max} - {r_min}) * 255 = {int(C[0, 0])}"
            ),
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "clipped": int(C[0, 0]),
            },
        },
    ]


def explain_hist_equalization(
    A: np.ndarray,
    C: np.ndarray,
    cdf_min: int,
    n: int,
) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Histogram",
            "expression": f"Computed histogram over 256 bins for N = {n} pixels",
            "latex": r"h(r) = \#\{x \in A : x = r\}",
            "values": {"N": int(n)},
        },
        {
            "step": 2,
            "title": "CDF",
            "expression": (
                f"cdf[r] = sum_{{k<=r}} h[k], cdf_min = {cdf_min}"
            ),
            "latex": r"\mathrm{CDF}(r) = \sum_{k=0}^{r} h(k)",
            "values": {"cdf_min": int(cdf_min)},
        },
        {
            "step": 3,
            "title": "Mapping",
            "expression": "s(r) = round((cdf(r) - cdf_min) / (N - cdf_min) * 255)",
            "latex": (
                r"s(r) = \mathrm{round}\!\left("
                r"\frac{\mathrm{CDF}(r) - \mathrm{CDF}_{min}}{N - \mathrm{CDF}_{min}}"
                r"\cdot 255\right)"
            ),
            "values": None,
        },
        {
            "step": 4,
            "title": "Spot check (top-left)",
            "expression": (
                f"pixel (0,0): r = {int(A[0, 0])} -> s = {int(C[0, 0])}"
            ),
            "latex": None,
            "values": {
                "A_00": int(A[0, 0]),
                "C_00": int(C[0, 0]),
            },
        },
    ]


def explain_sharpen(A: np.ndarray, strength: float, C: np.ndarray) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Parameter",
            "expression": f"strength alpha = {strength}",
            "latex": None,
            "values": {"strength": strength},
        },
        {
            "step": 2,
            "title": "Definition",
            "expression": "s = r + alpha * (r - blur(r)), with blur = Gaussian 3x3 sigma=1",
            "latex": r"s = r + \alpha\,(r - \mathrm{blur}(r))",
            "values": None,
        },
        {
            "step": 3,
            "title": "Note",
            "expression": (
                "The blur operator is applied via OpenCV. Interior pixel "
                "values are then combined linearly."
            ),
            "latex": None,
            "values": {"A_00": int(A[0, 0]), "C_00": int(C[0, 0])},
        },
    ]


def explain_convolution_pixel(trace: dict) -> list[dict]:
    """Turn a pixel trace dict (from pixel_trace.trace_convolution_pixel)
    into a readable step list. Every numeric value here comes directly
    from the trace — no fabrication.
    """
    mult = trace["multiplications"]
    kh, kw = trace["kernel_shape"]
    step_list: list[dict] = [
        {
            "step": 1,
            "title": "Kernel",
            "expression": f"{kh}x{kw} kernel; sum of weights = "
                          f"{sum(m['kernel_value'] for m in mult[:0] or [{'kernel_value': float(np.sum(trace['kernel_values']))}])}"[:120],
            "latex": None,
            "values": {"kernel_shape": [kh, kw]},
        },
        {
            "step": 2,
            "title": "Padding",
            "expression": f"Padding mode = '{trace['padding']}'",
            "latex": None,
            "values": {"padding": trace["padding"]},
        },
        {
            "step": 3,
            "title": "Local neighborhood",
            "expression": (
                f"Window {kh}x{kw} around pixel "
                f"({trace['pixel'][0]}, {trace['pixel'][1]})"
            ),
            "latex": None,
            "values": {"window": trace["neighborhood"]},
        },
        {
            "step": 4,
            "title": "Element-wise products",
            "expression": f"{len(mult)} multiplications and their products",
            "latex": None,
            "values": {"multiplications": mult},
        },
        {
            "step": 5,
            "title": "Summation",
            "expression": f"sum = {trace['sum']:.6f}",
            "latex": None,
            "values": {"sum": trace["sum"]},
        },
        {
            "step": 6,
            "title": "Result",
            "expression": f"output pixel value = {trace['result']:.6f}",
            "latex": None,
            "values": {"result": trace["result"]},
        },
    ]
    return step_list


def explain_sobel_gradient(
    gx: np.ndarray,
    gy: np.ndarray,
    magnitude: np.ndarray,
    row: int,
    col: int,
) -> list[dict]:
    return [
        {
            "step": 1,
            "title": "Kernels",
            "expression": (
                "Kx = [[-1,0,1],[-2,0,2],[-1,0,1]], "
                "Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]"
            ),
            "latex": None,
            "values": None,
        },
        {
            "step": 2,
            "title": "Convolutions",
            "expression": "Gx = I * Kx,  Gy = I * Ky",
            "latex": r"G_x = I * K_x,\quad G_y = I * K_y",
            "values": None,
        },
        {
            "step": 3,
            "title": "Magnitude",
            "expression": "|grad| = sqrt(Gx^2 + Gy^2)",
            "latex": r"|\nabla I| = \sqrt{G_x^2 + G_y^2}",
            "values": None,
        },
        {
            "step": 4,
            "title": "Spot check",
            "expression": (
                f"pixel ({row}, {col}): Gx = {float(gx[row, col]):.3f}, "
                f"Gy = {float(gy[row, col]):.3f}, "
                f"|grad| = {float(magnitude[row, col]):.3f}"
            ),
            "latex": None,
            "values": {
                "Gx": float(gx[row, col]),
                "Gy": float(gy[row, col]),
                "magnitude": float(magnitude[row, col]),
            },
        },
    ]
# ===========================================================================
# Phase 4 builders (covariance, eigen, PCA, SVD)
# ===========================================================================
def explain_covariance(X: np.ndarray, C: np.ndarray) -> list[dict]:
    N, D = X.shape
    return [
        {
            "step": 1,
            "title": "Data shape",
            "expression": f"X has shape {N} samples x {D} features",
            "latex": r"X \in \mathbb{R}^{N \times D}",
            "values": {"N": N, "D": D},
        },
        {
            "step": 2,
            "title": "Mean-center",
            "expression": "Z = X - mean(X, axis=0)",
            "latex": r"Z = X - \bar{X}",
            "values": None,
        },
        {
            "step": 3,
            "title": "Covariance",
            "expression": f"C has shape {C.shape} = {D} x {D}",
            "latex": r"C = \frac{1}{N-1} Z^{\top} Z",
            "values": {
                "shape": list(C.shape),
                "trace": float(np.trace(C)),
            },
        },
    ]


def explain_pca(
    eigenvalues: np.ndarray,
    evr: np.ndarray,
    cumulative: np.ndarray,
    k_used: int,
) -> list[dict]:
    top5 = [float(x) for x in eigenvalues[:min(5, eigenvalues.size)]]
    return [
        {
            "step": 1,
            "title": "Eigenvalues (top)",
            "expression": f"top-5 = {top5}",
            "latex": r"\lambda_1 \geq \lambda_2 \geq \dots",
            "values": {"top5": top5},
        },
        {
            "step": 2,
            "title": "Explained variance",
            "expression": (
                f"EVR_1 = {float(evr[0]):.4f}, "
                f"cumulative@{k_used} = {float(cumulative[k_used - 1]):.4f}"
            ),
            "latex": r"\mathrm{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}",
            "values": {
                "evr_1": float(evr[0]),
                "cumulative_at_k": float(cumulative[k_used - 1]),
            },
        },
        {
            "step": 3,
            "title": "Reconstruction",
            "expression": (
                f"X_hat = mean + coefficients @ components  "
                f"(using k = {k_used})"
            ),
            "latex": (
                r"\hat{X} = \bar{X} + A_k V_k^{\top}"
            ),
            "values": {"k": int(k_used)},
        },
    ]


def explain_svd(
    m: int,
    n: int,
    S: np.ndarray,
    k: int,
    rel_err: float,
    comp_ratio: float,
) -> list[dict]:
    top5 = [float(x) for x in S[:min(5, S.size)]]
    return [
        {
            "step": 1,
            "title": "Shape",
            "expression": f"A is {m} x {n}; k = {k}",
            "latex": r"A \in \mathbb{R}^{m \times n}",
            "values": {"m": m, "n": n, "k": k},
        },
        {
            "step": 2,
            "title": "Singular values (top)",
            "expression": f"top-5 = {top5}",
            "latex": r"\sigma_1 \geq \sigma_2 \geq \dots \geq 0",
            "values": {"top5": top5},
        },
        {
            "step": 3,
            "title": "Low-rank reconstruction",
            "expression": f"A_k = U_k Sigma_k V_k^T with k = {k}",
            "latex": r"A_k = U_k \Sigma_k V_k^{\top}",
            "values": {"k": int(k)},
        },
        {
            "step": 4,
            "title": "Error",
            "expression": f"relative Frobenius error = {rel_err:.6f}",
            "latex": r"\|A - A_k\|_F / \|A\|_F",
            "values": {"relative_error": float(rel_err)},
        },
        {
            "step": 5,
            "title": "Compression ratio",
            "expression": (
                f"full storage {m*n} floats vs rank-{k} storage "
                f"{k*(m+n+1)} floats -> {comp_ratio:.2f}x"
            ),
            "latex": None,
            "values": {"compression_ratio": float(comp_ratio)},
        },
    ]