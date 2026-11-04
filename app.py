"""Chest X-ray Mathematical Lab — Linear Algebra Image Processing.

Phase 1 entry point:
    - Application shell & navigation
    - X-ray upload with validation
    - Grayscale preprocessing
    - Live numerical inspection of the resulting matrix

Later phases plug in via the `pages/` directory and add the mathematical
trace engine UI.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import streamlit as st

from core.image_loader import (
    build_metadata,
    load_image,
    to_grayscale,
)
from core.image_preprocessing import ensure_grayscale, ensure_uint8


# ---------------------------------------------------------------------------
# Page configuration MUST be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chest X-ray Mathematical Lab",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Initialize keys we read/write throughout the app."""
    defaults = {
        "gray_image": None,       # np.ndarray (H, W) uint8
        "original_image": None,   # np.ndarray original
        "metadata": None,         # ImageMetadata
        "filename": None,         # str
        "operation_log": [],      # list[dict] — used from Phase 2 onward
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _render_matrix_preview(
    arr: np.ndarray,
    n_rows: int = 6,
    n_cols: int = 6,
) -> None:
    """Show a small top-left block of the array as a table."""
    h, w = arr.shape
    r = min(n_rows, h)
    c = min(n_cols, w)
    block = arr[:r, :c]

    st.caption(f"Top-left {r}x{c} block of the {h}x{w} matrix:")
    st.dataframe(
        block,
        use_container_width=False,
        height=min(40 + 35 * r, 300),
    )


def _render_sidebar_uploader() -> None:
    """Sidebar: upload + validate + grayscale."""
    with st.sidebar:
        st.markdown("## 📤 Upload Chest X-ray")
        uploaded = st.file_uploader(
            "PNG, JPG or JPEG · max 15 MB",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="main_uploader",
        )

        st.divider()
        st.markdown("### About")
        st.markdown(
            "An **educational** linear-algebra sandbox for chest X-ray "
            "image processing. This is **not** a medical diagnostic tool."
        )
        st.divider()
        st.caption("Phase 1 · Upload & Preprocess")

        if uploaded is not None:
            try:
                with st.spinner("Validating & decoding image…"):
                    t0 = time.perf_counter()
                    original = load_image(uploaded.getvalue(), uploaded.name)
                    gray = to_grayscale(original)
                    gray = ensure_grayscale(ensure_uint8(gray))
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    meta = build_metadata(original, gray)

                st.session_state.original_image = original
                st.session_state.gray_image = gray
                st.session_state.metadata = meta
                st.session_state.filename = uploaded.name

                st.success(
                    f"Loaded **{uploaded.name}** in {elapsed_ms:.1f} ms "
                    f"({gray.shape[1]}x{gray.shape[0]} px, uint8)."
                )
            except ValueError as exc:
                st.session_state.gray_image = None
                st.session_state.original_image = None
                st.session_state.metadata = None
                st.session_state.filename = None
                st.error(f"❌ {exc}")
            except Exception as exc:  # last-resort safety net
                st.session_state.gray_image = None
                st.session_state.original_image = None
                st.session_state.metadata = None
                st.session_state.filename = None
                st.error(f"❌ Unexpected error while loading image: {exc}")
                with st.expander("Show traceback (for developers)"):
                    st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
_render_sidebar_uploader()

st.title("🩻 Chest X-ray Mathematical Lab")
st.markdown(
    "**Linear Algebra-based Image Processing & Enhancement** · "
    "*Educational sandbox*"
)

st.info(
    "This application demonstrates **mathematical** linear-algebra techniques "
    "on chest X-ray images. It is **not** a medical device and must never be "
    "used for clinical diagnosis."
)

gray = st.session_state.gray_image
meta = st.session_state.metadata

# ---- Empty state ----
if gray is None:
    st.warning("⬅️ **Upload a chest X-ray in the sidebar** to begin.")
    st.markdown(
        """
        ### What you will be able to explore (roadmap)
        - **Module 1** · Matrix Explorer — pixels as numbers, histograms
        - **Module 2** · Enhancement — brightness, contrast, histogram eq, CLAHE
        - **Module 3** · Filtering — mean, Gaussian, median, custom convolution
        - **Module 4** · Edge Detection — Sobel, Laplacian, Canny
        - **Module 5** · PCA & Eigenvalues — eigenspaces, variance, reconstruction
        - **Module 6** · SVD Compression — rank-k approximation `A ≈ Uk·Σk·VkT`
        - **Module 7** · Feature Analysis — statistical descriptors
        - **Module 8** · Reports — MSE, PSNR, SSIM, exports

        Every operation will expose its **actual mathematical trace** —
        no fabricated numbers, no hidden library calls.
        """
    )
    st.stop()

# ---- Loaded state ----
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("🖼️ Image Workspace")
    tab_orig, tab_gray, tab_cmp = st.tabs(["Original", "Grayscale", "Compare"])
    original = st.session_state.original_image

    with tab_orig:
        if original.ndim == 3 and original.shape[2] >= 3:
            st.image(
                original,
                caption="Original color upload",
                use_column_width=True,
            )
        else:
            st.image(
                original,
                caption="Original grayscale upload",
                use_column_width=True,
                clamp=True,
            )

    with tab_gray:
        st.image(
            gray,
            caption="Grayscale (ITU-R BT.601 luma)",
            use_column_width=True,
            clamp=True,
        )

    with tab_cmp:
        c1, c2 = st.columns(2)
        c1.image(original, caption="Original", use_column_width=True, clamp=True)
        c2.image(gray, caption="Grayscale", use_column_width=True, clamp=True)

with right:
    st.subheader("🧮 Mathematical Workspace")
    st.caption("All values below are computed **from your uploaded image**.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Height (rows)", f"{meta.height}")
    m2.metric("Width (cols)", f"{meta.width}")
    m3.metric("Total pixels", f"{meta.height * meta.width:,}")

    m4, m5, m6 = st.columns(3)
    m4.metric("dtype", meta.array_dtype)
    m5.metric("Intensity range", f"{meta.intensity_min}–{meta.intensity_max}")
    m6.metric("Original mode", meta.original_mode)

    st.markdown("#### Numerical summary")
    stats_table = {
        "Statistic": [
            "Mean",
            "Median",
            "Std. deviation",
            "Variance",
            "Minimum",
            "Maximum",
        ],
        "Value": [
            f"{meta.intensity_mean:.3f}",
            f"{meta.intensity_median:.3f}",
            f"{meta.intensity_std:.3f}",
            f"{meta.intensity_std ** 2:.3f}",
            f"{meta.intensity_min}",
            f"{meta.intensity_max}",
        ],
    }
    st.table(stats_table)

    with st.expander("🔍 Matrix preview (top-left block)", expanded=False):
        _render_matrix_preview(gray)

    with st.expander("ℹ️ Formula used for grayscale conversion", expanded=False):
        st.latex(r"Y = 0.299\,R + 0.587\,G + 0.114\,B")
        st.caption(
            "ITU-R BT.601 luma coefficients, identical to OpenCV's "
            "`cvtColor(..., COLOR_BGR2GRAY)` path."
        )

st.divider()
st.caption(
    f"File: `{st.session_state.filename}` · "
    f"Matrix shape `{gray.shape}` · dtype `{gray.dtype}`"
)