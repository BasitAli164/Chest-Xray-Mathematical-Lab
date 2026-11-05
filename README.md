# 🖼️ Image Mathematical Lab

An **educational** Streamlit web application that demonstrates **Linear Algebra
and mathematical image-processing algorithms** on digital images. 

> ⚠️ **Not a medical device.** No pre-trained AI models for image processing, 
> no diagnostic claims. Every operation is a deterministic linear-algebra algorithm.
> Never use for clinical decision-making.

## 🌟 Highlights

- **No Pre-trained AI Models:** Image processing relies strictly on deterministic mathematics (NumPy, SciPy, OpenCV), making it perfect for learning.
- **Full Mathematical Transparency:** Every operation records the *actual* formula, step-by-step derivation, and (where applicable) pixel-level traces. No fabricated numbers.
- **Three-Level Mathematical Workspace:** Formula (Level 1) → Step-by-step Derivation (Level 2) → Deep Pixel-level Numerical Inspection (Level 3).
- **Generative AI Explainer:** Integrated with Groq API (LLaMA-3) to explain complex mathematical results in simple, non-technical language without hallucinating data.
- **Side-by-Side Layout:** Left panel displays the image comparison, right panel displays the live math workspace.
- **Fully Tested:** Over 240+ automated pytest cases ensuring mathematical correctness.

## 🧩 Modules

| Module | What it does |
|--------|--------------|
| **Dashboard** | View session history, total operations performed, and quick links. |
| **Matrix Explorer** | Treat pixels as numbers, explore transpose, addition, Hadamard product, and row/column inspections. |
| **Enhancement** | Brightness, contrast, min-max normalization, histogram equalization, CLAHE, and sharpening. |
| **Noise Reduction** | Mean, Gaussian, and median filtering. Includes custom kernel builder with manual + optimized convolution comparisons. |
| **Edge Detection** | Sobel X/Y, gradient magnitude, Laplacian, and Canny edge detection. |
| **PCA Analysis** | Dual PCA over a reference stack to demonstrate dimensionality reduction, eigenvalues, and variance. |
| **SVD Compression** | Interactive rank-k approximation with singular value distributions and compression ratio. |
| **Feature Analysis** | Extraction of intensity, gradient, edge density, histogram entropy, and optional GLCM texture features. |
| **Reports** | Generate MSE, PSNR, and SSIM metrics, plus downloadable CSV data and PDF summary reports. |
| **🤖 AI Explainer** | Available on every operation, providing a 6-point easy-to-understand explanation (Concept, Math, Visual Impact, Parameters, Use Case, Sequence). |

## 🚀 Quick Start (Local Development)

Follow these steps to run the application on your local machine using Miniconda:

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Chest-Xray-Mathematical-Lab.git
cd Chest-Xray-Mathematical-Lab

# 2. Create and activate the conda environment
conda create -n xraylab python=3.11 -y
conda activate xraylab

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt