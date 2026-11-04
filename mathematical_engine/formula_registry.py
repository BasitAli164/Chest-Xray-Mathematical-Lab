"""Central registry of formulas used by every operation.

Each entry has:
    - `name`: human-readable operation name
    - `latex`: LaTeX for st.latex()
    - `plain`: plain text for logs / CSV reports
    - `description`: short explanation

Add new formulas here — never inline them in the UI or core modules.
"""

from __future__ import annotations

from typing import TypedDict


class FormulaEntry(TypedDict):
    name: str
    latex: str
    plain: str
    description: str


FORMULAS: dict[str, FormulaEntry] = {
    # ---- Matrix operations ------------------------------------------------
    "transpose": {
        "name": "Transpose",
        "latex": r"A^{\top}_{j,i} = A_{i,j}",
        "plain": "A^T[j, i] = A[i, j]",
        "description": (
            "The transpose flips rows and columns. An (m x n) matrix becomes "
            "(n x m)."
        ),
    },
    "add": {
        "name": "Matrix Addition",
        "latex": r"C_{i,j} = A_{i,j} + B_{i,j}",
        "plain": "C[i, j] = A[i, j] + B[i, j]",
        "description": "Element-wise sum of two equal-shape matrices.",
    },
    "subtract": {
        "name": "Matrix Subtraction",
        "latex": r"C_{i,j} = A_{i,j} - B_{i,j}",
        "plain": "C[i, j] = A[i, j] - B[i, j]",
        "description": "Element-wise difference of two equal-shape matrices.",
    },
    "scalar_multiply": {
        "name": "Scalar Multiplication",
        "latex": r"C_{i,j} = k \cdot A_{i,j}",
        "plain": "C[i, j] = k * A[i, j]",
        "description": "Multiplies every entry of A by the scalar k.",
    },
    "elementwise_multiply": {
        "name": "Hadamard Product",
        "latex": r"C_{i,j} = A_{i,j} \odot B_{i,j}",
        "plain": "C[i, j] = A[i, j] * B[i, j]",
        "description": "Element-wise product (not the matrix product).",
    },
    "matrix_multiply": {
        "name": "Matrix Product",
        "latex": r"C_{i,j} = \sum_{k=0}^{n-1} A_{i,k} \, B_{k,j}",
        "plain": "C[i, j] = sum_k A[i, k] * B[k, j]",
        "description": (
            "Classical matrix product. If A is (m x n) and B is (n x p) "
            "then C is (m x p)."
        ),
    },
    # ---- Statistics --------------------------------------------------------
    "mean": {
        "name": "Mean",
        "latex": r"\mu = \frac{1}{N}\sum_{i=1}^{N} x_i",
        "plain": "mu = (1/N) * sum(x_i)",
        "description": "Arithmetic average of all pixel values.",
    },
    "variance": {
        "name": "Variance",
        "latex": r"\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2",
        "plain": "sigma^2 = (1/N) * sum((x_i - mu)^2)",
        "description": "Average squared deviation from the mean.",
    },
    "std": {
        "name": "Standard Deviation",
        "latex": r"\sigma = \sqrt{\sigma^2}",
        "plain": "sigma = sqrt(variance)",
        "description": "Square root of the variance.",
    },
    "frobenius": {
        "name": "Frobenius Norm",
        "latex": r"\|A\|_F = \sqrt{\sum_{i,j} A_{i,j}^2}",
        "plain": "||A||_F = sqrt(sum(A[i, j]^2))",
        "description": "Euclidean norm of the matrix viewed as a flat vector.",
    },
    # ---- Intensity transforms (Phase 2) -----------------------------------
    "identity": {
        "name": "Identity Transform",
        "latex": r"s = r",
        "plain": "s = r",
        "description": "No-op; useful as a baseline.",
    },
    "linear": {
        "name": "Linear Scaling",
        "latex": r"s = g \cdot r + b",
        "plain": "s = g * r + b",
        "description": "Affine transform with gain g and offset b.",
    },
    "negative": {
        "name": "Negative",
        "latex": r"s = 255 - r",
        "plain": "s = 255 - r",
        "description": "Inverts intensities about the midpoint.",
    },
    "log": {
        "name": "Logarithmic",
        "latex": r"s = c \cdot \ln(1 + r)",
        "plain": "s = c * ln(1 + r)",
        "description": "Compresses bright regions, expands dark ones.",
    },
    "gamma": {
        "name": "Gamma",
        "latex": r"s = 255 \left(\frac{r}{255}\right)^{\gamma}",
        "plain": "s = 255 * (r / 255) ^ gamma",
        "description": "gamma < 1 brightens; gamma > 1 darkens.",
    },
    "contrast_stretch": {
        "name": "Contrast Stretch",
        "latex": r"s = \frac{r - r_{min}}{r_{max} - r_{min}} \cdot 255",
        "plain": "s = (r - min) / (max - min) * 255",
        "description": "Min-max normalization to full 8-bit range.",
    },
    # ---- Enhancement (Phase 3) --------------------------------------------
    "brightness": {
        "name": "Brightness Adjustment",
        "latex": r"s = \mathrm{clip}(r + b,\, 0,\, 255)",
        "plain": "s = clip(r + b, 0, 255)",
        "description": "Adds a constant offset to every pixel value.",
    },
    "contrast": {
        "name": "Contrast Adjustment",
        "latex": r"s = \mathrm{clip}(g \cdot (r - 128) + 128,\, 0,\, 255)",
        "plain": "s = clip(g * (r - 128) + 128, 0, 255)",
        "description": (
            "Scales intensity deviations around the midpoint 128."
        ),
    },
    "min_max_normalize": {
        "name": "Min-Max Normalization",
        "latex": r"s = \frac{r - r_{min}}{r_{max} - r_{min}} \cdot 255",
        "plain": "s = (r - min) / (max - min) * 255",
        "description": "Stretches intensities to the full [0, 255] range.",
    },
    "histogram_equalization": {
        "name": "Histogram Equalization",
        "latex": (
            r"s(r) = \mathrm{round}\!\left("
            r"\frac{\mathrm{CDF}(r) - \mathrm{CDF}_{min}}{N - \mathrm{CDF}_{min}}"
            r"\cdot 255\right)"
        ),
        "plain": "s(r) = round((CDF(r) - CDF_min) / (N - CDF_min) * 255)",
        "description": (
            "Redistributes intensities so the CDF becomes approximately "
            "linear."
        ),
    },
    "clahe": {
        "name": "CLAHE",
        "latex": r"s = \mathrm{CLAHE}(r;\ c,\ g)",
        "plain": "s = CLAHE(r; clip_limit, tile_grid)",
        "description": (
            "Contrast Limited Adaptive Histogram Equalization. Computes "
            "local histograms over a grid and clips contrast."
        ),
    },
    "sharpen": {
        "name": "Unsharp Mask Sharpening",
        "latex": r"s = r + \alpha \cdot (r - \mathrm{blur}(r))",
        "plain": "s = r + alpha * (r - blur(r))",
        "description": (
            "Adds a scaled version of the high-pass (detail) signal back "
            "to the original."
        ),
    },
    # ---- Convolution and filtering (Phase 3) ------------------------------
    "convolution": {
        "name": "2D Convolution",
        "latex": r"O(i,j) = \sum_u \sum_v I(i-u,\, j-v) \, K(u,v)",
        "plain": "O(i,j) = sum_u sum_v I(i-u, j-v) * K(u,v)",
        "description": (
            "Sliding-window weighted sum. Used for blurring, sharpening "
            "and edge detection."
        ),
    },
    "mean_filter": {
        "name": "Mean Filter",
        "latex": r"O(i,j) = \frac{1}{k^2}\sum_{u,v} I(i+u, j+v)",
        "plain": "O(i,j) = (1/k^2) * sum I(i+u, j+v)",
        "description": "Uniform blur; every neighbor weighted equally.",
    },
    "gaussian_filter": {
        "name": "Gaussian Filter",
        "latex": (
            r"K(u,v) = \frac{1}{2\pi\sigma^2}"
            r"\exp\!\left(-\frac{u^2 + v^2}{2\sigma^2}\right)"
        ),
        "plain": "K(u,v) = (1/(2*pi*sigma^2)) * exp(-(u^2+v^2)/(2*sigma^2))",
        "description": "Weighted blur with Gaussian weights.",
    },
    "median_filter": {
        "name": "Median Filter",
        "latex": r"O(i,j) = \mathrm{median}\{ I(i+u, j+v) \}",
        "plain": "O(i,j) = median({I(i+u, j+v)})",
        "description": (
            "Non-linear filter; replaces each pixel with the median of its "
            "neighborhood. Excellent at removing salt-and-pepper noise."
        ),
    },
    # ---- Edge detection (Phase 3) -----------------------------------------
    "sobel_x": {
        "name": "Sobel X",
        "latex": r"G_x = I * K_x,\quad K_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}",
        "plain": "Gx = I * [[-1,0,1],[-2,0,2],[-1,0,1]]",
        "description": "Horizontal derivative; highlights vertical edges.",
    },
    "sobel_y": {
        "name": "Sobel Y",
        "latex": r"G_y = I * K_y,\quad K_y = \begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}",
        "plain": "Gy = I * [[-1,-2,-1],[0,0,0],[1,2,1]]",
        "description": "Vertical derivative; highlights horizontal edges.",
    },
    "gradient_magnitude": {
        "name": "Gradient Magnitude",
        "latex": r"|\nabla I| = \sqrt{G_x^2 + G_y^2}",
        "plain": "|grad I| = sqrt(Gx^2 + Gy^2)",
        "description": "Combines X and Y derivatives into a single edge-strength map.",
    },
    "laplacian": {
        "name": "Laplacian",
        "latex": r"\nabla^2 I = I * L,\quad L = \begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}",
        "plain": "Laplacian = I * [[0,1,0],[1,-4,1],[0,1,0]]",
        "description": "Second derivative; detects edges and corners.",
    },
    "canny": {
        "name": "Canny Edge Detection",
        "latex": r"\mathrm{Canny}(I;\ t_{low}, t_{high})",
        "plain": "Canny(I; low, high)",
        "description": (
            "Multi-stage edge detector (Gaussian smoothing, gradient, "
            "NMS, hysteresis). Internal stages not individually exposed."
        ),
    },
        # ---- Phase 4: Eigenanalysis, PCA, SVD -------------------------------
    "covariance": {
        "name": "Covariance Matrix",
        "latex": r"C = \frac{1}{N-1} Z^{\top} Z,\quad Z = X - \bar{X}",
        "plain": "C = (1/(N-1)) * Z^T Z,  Z = X - mean(X)",
        "description": (
            "Sample covariance of mean-centered data. Diagonal = variances, "
            "off-diagonal = covariances."
        ),
    },
    "eigen_decomposition": {
        "name": "Eigen-decomposition",
        "latex": r"C v = \lambda v",
        "plain": "C v = lambda v",
        "description": (
            "Eigenvectors are directions unchanged by C; eigenvalues are "
            "the scaling factors. For symmetric C, eigenvectors form an "
            "orthonormal basis."
        ),
    },
    "pca": {
        "name": "Principal Component Analysis",
        "latex": (
            r"X \approx \bar{X} + \sum_{i=1}^{k} a_i v_i^{\top},"
            r"\quad v_i = \text{eigenvectors of } C"
        ),
        "plain": "X_hat = mean + sum_{i=1..k} a_i * v_i",
        "description": (
            "Projects data onto the top-k eigenvectors of the covariance "
            "matrix, capturing the most variance."
        ),
    },
    "explained_variance": {
        "name": "Explained Variance Ratio",
        "latex": (
            r"\mathrm{EVR}_i = "
            r"\frac{\lambda_i}{\sum_{j=1}^{M} \lambda_j}"
        ),
        "plain": "EVR_i = lambda_i / sum(lambda_j)",
        "description": (
            "Fraction of total variance captured by the i-th principal "
            "component."
        ),
    },
    "svd": {
        "name": "Singular Value Decomposition",
        "latex": r"A = U \Sigma V^{\top}",
        "plain": "A = U * Sigma * V^T",
        "description": (
            "Every matrix factors into orthogonal U, diagonal Sigma "
            "(singular values), and orthogonal V."
        ),
    },
    "svd_rank_k": {
        "name": "Rank-k Approximation",
        "latex": r"A_k = U_k \Sigma_k V_k^{\top}",
        "plain": "A_k = U_k * Sigma_k * V_k^T",
        "description": (
            "Optimal rank-k approximation (Eckart-Young theorem): "
            "minimizes ||A - A_k||_F."
        ),
    },
    "svd_error": {
        "name": "Reconstruction Error",
        "latex": r"\frac{\|A - A_k\|_F}{\|A\|_F}",
        "plain": "||A - A_k||_F / ||A||_F",
        "description": (
            "Relative Frobenius error. Also equals sqrt(sum of tail "
            "singular values squared / total energy squared)."
        ),
    },
        # ---- Phase 6: Metrics & Feature Statistics --------------------------
    "mse": {
        "name": "Mean Squared Error",
        "latex": r"\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(a_i - b_i)^2",
        "plain": "MSE = (1/N) * sum((a_i - b_i)^2)",
        "description": (
            "Average of squared pixel differences. A mathematical "
            "difference measure, not a quality claim."
        ),
    },
    "psnr": {
        "name": "Peak Signal-to-Noise Ratio",
        "latex": r"\mathrm{PSNR} = 10 \log_{10}\!\left(\frac{R^2}{\mathrm{MSE}}\right)",
        "plain": "PSNR = 10 * log10(R^2 / MSE)",
        "description": (
            "Log-scale measure of reconstruction fidelity. R is the "
            "data range (255 for 8-bit images). Higher is 'more similar', "
            "not 'better diagnostically'."
        ),
    },
    "ssim": {
        "name": "Structural Similarity Index",
        "latex": (
            r"\mathrm{SSIM}(x,y) = "
            r"\frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}"
            r"{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}"
        ),
        "plain": "SSIM = structural similarity in [0, 1]",
        "description": (
            "Measures luminance, contrast, and structure agreement between "
            "two images. 1 = identical, 0 = unrelated."
        ),
    },
    "skewness": {
        "name": "Skewness",
        "latex": r"\gamma_1 = \frac{1}{N}\sum_{i}\left(\frac{x_i - \mu}{\sigma}\right)^3",
        "plain": "skew = (1/N) * sum(((x - mu) / sigma)^3)",
        "description": "Third standardized moment — asymmetry of the histogram.",
    },
    "kurtosis": {
        "name": "Excess Kurtosis",
        "latex": r"\kappa = \frac{1}{N}\sum_{i}\left(\frac{x_i - \mu}{\sigma}\right)^4 - 3",
        "plain": "kurt = (1/N) * sum(((x - mu) / sigma)^4) - 3",
        "description": "Fourth standardized moment minus 3 — histogram tailedness.",
    },
    "entropy": {
        "name": "Shannon Entropy",
        "latex": r"H = -\sum_{i} p_i \log_2 p_i",
        "plain": "H = -sum(p_i * log2(p_i))",
        "description": (
            "Information content of the intensity histogram. Higher = more "
            "spread-out intensity distribution."
        ),
    },
    "glcm": {
        "name": "GLCM Texture",
        "latex": r"\mathrm{GLCM}(i,j; d, \theta)",
        "plain": "GLCM properties at distance d and angle theta",
        "description": (
            "Co-occurrence statistics of neighboring pixel intensities. "
            "Used for texture — computed via scikit-image (internals not "
            "individually traced)."
        ),
    },
}


def get_formula(key: str) -> FormulaEntry:
    """Return a FormulaEntry or raise KeyError with a helpful message."""
    if key not in FORMULAS:
        raise KeyError(
            f"Unknown formula key '{key}'. "
            f"Known keys: {sorted(FORMULAS.keys())}"
        )
    return FORMULAS[key]