"""Enhancement page — brightness, contrast, normalization, hist-eq, CLAHE,
sharpening. Side-by-side image + mathematical workspace.
"""

from __future__ import annotations

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core.enhancement import (
    adjust_brightness,
    adjust_contrast,
    apply_clahe,
    compute_histogram,
    histogram_equalization_manual,
    histogram_equalization_opencv,
    min_max_normalize,
    sharpen,
)
from mathematical_engine.explanation_templates import (
    explain_brightness,
    explain_contrast,
    explain_hist_equalization,
    explain_min_max,
    explain_sharpen,
)
from mathematical_engine.numerical_calculator import maximum_abs_difference
from mathematical_engine.trace_manager import TraceManager


st.set_page_config(page_title="Enhancement · Chest X-ray Lab", page_icon="✨",
                   layout="wide")

if "trace_manager" not in st.session_state:
    st.session_state.trace_manager = TraceManager()

gray = st.session_state.get("gray_image")
tm: TraceManager = st.session_state.trace_manager

st.title("✨ Image Enhancement")
st.caption(
    "Point-wise intensity transformations. Every value below is computed "
    "from your uploaded image."
)

if gray is None:
    st.warning("⬅️ Upload a chest X-ray on the main page first.")
    st.stop()


# ---------------------------------------------------------------------------
# Operation selector
# ---------------------------------------------------------------------------
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.subheader("🖼️ Preview")
    op = st.selectbox(
        "Operation",
        options=[
            "Brightness",
            "Contrast",
            "Min-Max Normalize",
            "Histogram Equalization",
            "CLAHE",
            "Sharpen",
        ],
        key="enh_op",
    )

    # Default parameters per op
    if op == "Brightness":
        offset = st.slider("offset b", min_value=-128, max_value=128,
                           value=30, step=1, key="enh_b")
    elif op == "Contrast":
        gain = st.slider("gain g", min_value=0.1, max_value=3.0,
                         value=1.5, step=0.05, key="enh_g")
    elif op == "CLAHE":
        clip_limit = st.slider("clip limit", min_value=0.5, max_value=8.0,
                               value=2.0, step=0.1, key="enh_cl")
        tile_grid = st.selectbox("tile grid", options=[4, 8, 16], index=1,
                                 key="enh_tg")
    elif op == "Sharpen":
        strength = st.slider("strength alpha", min_value=0.0, max_value=3.0,
                             value=1.0, step=0.1, key="enh_sh")

    run = st.button("Apply", key="enh_run", type="primary")

    # Always show original
    tab_orig, tab_out, tab_cmp = st.tabs(["Original", "Processed", "Compare"])
    with tab_orig:
        st.image(gray, caption=f"Original {gray.shape}",
                 use_column_width=True, clamp=True)


# ---------------------------------------------------------------------------
# Run + store result in session for the right panel
# ---------------------------------------------------------------------------
result_image = None
result_derivation = None
result_summary = None

if run:
    try:
        t0 = time.perf_counter()
        if op == "Brightness":
            result_image = adjust_brightness(gray, offset=int(offset))
            derivation = explain_brightness(gray, int(offset), result_image)
            params = {"offset": int(offset)}
            formula_key = "brightness"
        elif op == "Contrast":
            result_image = adjust_contrast(gray, gain=float(gain))
            derivation = explain_contrast(gray, float(gain), result_image)
            params = {"gain": float(gain)}
            formula_key = "contrast"
        elif op == "Min-Max Normalize":
            result_image = min_max_normalize(gray)
            derivation = explain_min_max(gray, result_image)
            params = {}
            formula_key = "min_max_normalize"
        elif op == "Histogram Equalization":
            result_image = histogram_equalization_manual(gray)
            # Compute cdf_min for the derivation
            flat = gray.ravel()
            hist = np.bincount(flat, minlength=256)
            cdf = np.cumsum(hist)
            cdf_min = int(cdf[cdf > 0][0]) if np.any(cdf > 0) else 0
            derivation = explain_hist_equalization(
                gray, result_image, cdf_min, flat.size
            )
            params = {"method": "manual (CDF)"}
            formula_key = "histogram_equalization"
        elif op == "CLAHE":
            result_image = apply_clahe(
                gray, clip_limit=float(clip_limit),
                tile_grid_size=(int(tile_grid), int(tile_grid)),
            )
            derivation = [{
                "step": 1,
                "title": "Algorithm",
                "expression": (
                    "CLAHE = Contrast Limited AHE. Local histograms on a "
                    f"{int(tile_grid)}x{int(tile_grid)} grid, contrast "
                    f"clipped at {float(clip_limit)}."
                ),
                "latex": r"s = \mathrm{CLAHE}(r;\ c,\ g)",
                "values": {"clip_limit": float(clip_limit),
                           "tile_grid": int(tile_grid)},
            }, {
                "step": 2,
                "title": "Note",
                "expression": (
                    "OpenCV runs the internal tile loop. Only the final "
                    "output is exposed to the trace — the per-tile "
                    "histograms are not individually visible."
                ),
                "latex": None,
                "values": None,
            }, {
                "step": 3,
                "title": "Spot check (top-left)",
                "expression": (
                    f"pixel(0,0): {int(gray[0, 0])} -> {int(result_image[0, 0])}"
                ),
                "latex": None,
                "values": {"A_00": int(gray[0, 0]),
                           "C_00": int(result_image[0, 0])},
            }]
            params = {"clip_limit": float(clip_limit),
                      "tile_grid": int(tile_grid)}
            formula_key = "clahe"
        else:  # Sharpen
            result_image = sharpen(gray, strength=float(strength))
            derivation = explain_sharpen(gray, float(strength), result_image)
            params = {"strength": float(strength)}
            formula_key = "sharpen"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        tm.record(
            operation=formula_key,
            input_array=gray,
            parameters=params,
            derivation=derivation,
            output=result_image,
            execution_time_ms=elapsed_ms,
            validation={"ok": True, "warnings": [], "errors": []},
        )

        # Store in session for right-panel rendering
        st.session_state["enh_result_image"] = result_image
        st.session_state["enh_result_derivation"] = derivation
        st.session_state["enh_result_params"] = params
        st.session_state["enh_result_op"] = op
        st.session_state["enh_result_ms"] = elapsed_ms

    except Exception as exc:
        st.error(f"❌ Enhancement failed: {exc}")
        st.code(traceback.format_exc())
        result_image = None


# ---------------------------------------------------------------------------
# Retrieve stored result (survives rerun after Apply)
# ---------------------------------------------------------------------------
stored_image = st.session_state.get("enh_result_image")
stored_derivation = st.session_state.get("enh_result_derivation")
stored_params = st.session_state.get("enh_result_params")
stored_op = st.session_state.get("enh_result_op")
stored_ms = st.session_state.get("enh_result_ms", 0.0)


# ---------------------------------------------------------------------------
# LEFT panel: show processed + compare
# ---------------------------------------------------------------------------
with left:
    if stored_image is not None:
        with tab_out:
            st.image(stored_image,
                     caption=f"{stored_op} · shape {stored_image.shape}",
                     use_column_width=True, clamp=True)
        with tab_cmp:
            c1, c2 = st.columns(2)
            c1.image(gray, caption="Original",
                     use_column_width=True, clamp=True)
            c2.image(stored_image, caption=stored_op,
                     use_column_width=True, clamp=True)

        # Histograms side by side
        st.markdown("#### Histogram comparison")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].hist(gray.ravel(), bins=64, range=(0, 256), color="#555")
        axes[0].set_title("Original")
        axes[0].set_xlabel("Intensity")
        axes[0].set_ylabel("Count")
        axes[1].hist(stored_image.ravel(), bins=64, range=(0, 256),
                     color="#0B6E99")
        axes[1].set_title(stored_op)
        axes[1].set_xlabel("Intensity")
        axes[1].set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ---------------------------------------------------------------------------
# RIGHT panel: mathematical workspace
# ---------------------------------------------------------------------------
with right:
    st.subheader("🧮 Mathematical workspace")
    if stored_image is None:
        st.info(
            "Choose an operation on the left, adjust parameters, then click "
            "**Apply** to compute and see the math here."
        )
    else:
        st.markdown(
            f"**Operation:** {stored_op}  \n"
            f"**Params:** `{stored_params}`  \n"
            f"**Time:** {stored_ms:.2f} ms"
        )

        with st.expander("📐 Formula", expanded=True):
            if stored_op == "Brightness":
                st.latex(r"s = \mathrm{clip}(r + b,\, 0,\, 255)")
            elif stored_op == "Contrast":
                st.latex(r"s = \mathrm{clip}(g\,(r - 128) + 128,\, 0,\, 255)")
            elif stored_op == "Min-Max Normalize":
                st.latex(
                    r"s = \frac{r - r_{min}}{r_{max} - r_{min}}\cdot 255"
                )
            elif stored_op == "Histogram Equalization":
                st.latex(
                    r"s(r) = \mathrm{round}\!\left("
                    r"\frac{\mathrm{CDF}(r) - \mathrm{CDF}_{min}}"
                    r"{N - \mathrm{CDF}_{min}}\cdot 255\right)"
                )
            elif stored_op == "CLAHE":
                st.latex(r"s = \mathrm{CLAHE}(r;\ c,\ g)")
            elif stored_op == "Sharpen":
                st.latex(r"s = r + \alpha\,(r - \mathrm{blur}(r))")

        with st.expander("🧾 Step-by-step derivation", expanded=True):
            if stored_derivation:
                for step in stored_derivation:
                    st.markdown(
                        f"**Step {step['step']} — {step['title']}**"
                    )
                    st.code(step["expression"], language="text")
                    if step.get("latex"):
                        st.latex(step["latex"])
                    if step.get("values"):
                        st.json(step["values"])

        with st.expander("📊 Before/After statistics", expanded=False):
            stats = {
                "Original": {
                    "mean": float(gray.mean()),
                    "std": float(gray.std()),
                    "min": int(gray.min()),
                    "max": int(gray.max()),
                },
                "Processed": {
                    "mean": float(stored_image.mean()),
                    "std": float(stored_image.std()),
                    "min": int(stored_image.min()),
                    "max": int(stored_image.max()),
                },
            }
            st.json(stats)
            diff = maximum_abs_difference(
                gray.astype(np.float64),
                stored_image.astype(np.float64),
            )
            st.caption(f"Max absolute pixel difference: {diff:.3f}")

        with st.expander("🔍 Selected pixel inspection", expanded=False):
            r = st.number_input(
                "row", min_value=0, max_value=gray.shape[0] - 1,
                value=gray.shape[0] // 2, step=1, key="enh_px_r",
            )
            c = st.number_input(
                "col", min_value=0, max_value=gray.shape[1] - 1,
                value=gray.shape[1] // 2, step=1, key="enh_px_c",
            )
            ov = int(gray[int(r), int(c)])
            pv = int(stored_image[int(r), int(c)])
            st.markdown(
                f"Pixel ({int(r)}, {int(c)}):  \n"
                f"- Original = **{ov}**  \n"
                f"- Processed = **{pv}**  \n"
                f"- Difference = **{pv - ov:+d}**"
            )

        # Recompute-check: manual vs OpenCV hist eq
        if stored_op == "Histogram Equalization":
            with st.expander("🔎 Manual vs OpenCV comparison", expanded=False):
                cv_result = histogram_equalization_opencv(gray)
                max_diff = maximum_abs_difference(
                    stored_image.astype(np.float64),
                    cv_result.astype(np.float64),
                )
                st.markdown(
                    f"Max abs difference between manual and OpenCV: "
                    f"**{max_diff:.0f}** intensity levels"
                )
                st.caption(
                    "Small differences (±1–2) are expected due to rounding. "
                    "Larger differences indicate different CDF conventions."
                )
                c1, c2 = st.columns(2)
                c1.image(stored_image, caption="Manual", use_column_width=True,
                         clamp=True)
                c2.image(cv_result, caption="OpenCV", use_column_width=True,
                         clamp=True)


# ---------------------------------------------------------------------------
# Recent traces
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 🧾 Recent enhancement computations")
recent = [t for t in tm.traces if t.operation in {
    "brightness", "contrast", "min_max_normalize",
    "histogram_equalization", "clahe", "sharpen",
}][-5:]
if not recent:
    st.caption("No enhancement operations yet in this session.")
else:
    for t in reversed(recent):
        with st.expander(
            f"{t.operation_id} · {t.operation} · {t.execution_time_ms:.2f} ms"
        ):
            st.json(t.to_dict())