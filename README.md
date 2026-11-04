# 🩻 Chest X-ray Mathematical Lab

An **educational** Streamlit application that demonstrates **linear algebra
and mathematical image-processing algorithms** on chest X-ray images.

> ⚠️ **Not a medical device.** No AI models, no diagnostic claims.
> Every operation is a deterministic linear-algebra algorithm.
> Never use for clinical decision-making.

## Highlights

- **No pre-trained models.** Only deterministic mathematics.
- **Full transparency.** Every operation records the *actual* formula,
  step-by-step derivation, and (where applicable) pixel-level trace.
- **Three-level mathematical workspace.** Formula → derivation → deep
  numerical inspection.
- **Complete toolkit:** matrix ops, enhancement, filtering, convolution,
  edge detection, PCA, SVD compression, feature extraction, quality metrics.
- **Side-by-side layout.** Left panel = images, right panel = math.
- **Fully tested.** 240+ pytest tests.

## Modules

| Module | What it does |
|--------|--------------|
| **Matrix Explorer** | Pixels as numbers, transpose, add, Hadamard, row/column inspection |
| **Enhancement** | Brightness, contrast, min-max, histogram equalization, CLAHE, sharpen |
| **Noise Reduction** | Mean, Gaussian, median, custom kernel convolution (manual + optimized) |
| **Edge Detection** | Sobel X/Y, gradient magnitude, Laplacian, Canny |
| **PCA Analysis** | Dual PCA over a reference stack, eigenvalue residual validation |
| **SVD Compression** | Rank-k approximation with Frobenius error and compression ratio |
| **Feature Analysis** | Intensity, gradient, edge density, histogram entropy, optional GLCM |
| **Reports** | Image comparison, MSE / PSNR / SSIM, downloadable CSV + PDF |

## Quick Start (Miniconda)

```bash
git clone https://github.com/<your-username>/Chest-Xray-Mathematical-Lab.git
cd Chest-Xray-Mathematical-Lab

conda create -n xraylab python=3.11 -y
conda activate xraylab
pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py