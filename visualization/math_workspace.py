"""Reusable Streamlit math-workspace renderer.

Provides a single entry point `render(trace, key_prefix)` that
displays the three transparency levels in tabs:

    - Level 1 — Formula & Overview
    - Level 2 — Step-by-step derivation (with ◀ ▶ navigation)
    - Level 3 — Deep numerical inspection (matrix preview, pixel trace)

All data comes from a Trace dict produced by TraceManager. This module
never recomputes values.

Usage in a page
---------------
    from visualization.math_workspace import render
    render(trace.to_dict(), key_prefix="enhancement")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from mathematical_engine.level_renderer import (
    build_level1,
    build_level2,
    build_level3,
    has_level3,
)
from mathematical_engine.step_controller import StepController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_controller(
    key_prefix: str,
    steps: list[dict[str, Any]],
) -> StepController:
    """Return a persisted StepController for the current operation."""
    ss_key = f"{key_prefix}_step_controller"
    steps_key = f"{key_prefix}_step_steps"

    # Reset controller if the underlying steps changed (new operation).
    if ss_key not in st.session_state or st.session_state.get(steps_key) != steps:
        ctrl = StepController(steps)
        ctrl.reset()                        # start at step 0
        st.session_state[ss_key] = ctrl
        st.session_state[steps_key] = steps
    return st.session_state[ss_key]


# ---------------------------------------------------------------------------
# Level 1
# ---------------------------------------------------------------------------
def _render_level1(trace: dict[str, Any]) -> None:
    lv1 = build_level1(trace)

    st.markdown(
        f"**Operation:** `{lv1['operation']}`  \n"
        f"**ID:** `{lv1['operation_id']}`  \n"
        f"**Time:** {lv1['execution_time_ms']:.3f} ms  \n"
        f"**Timestamp:** {lv1['timestamp']}"
    )

    st.markdown("**Formula:**")
    if lv1["formula_latex"]:
        st.latex(lv1["formula_latex"])
    if lv1["description"]:
        st.caption(lv1["description"])

    c1, c2 = st.columns(2)
    c1.markdown(
        f"**Input shape**  \n"
        f"`{lv1['input_shape']}`  \n"
        f"dtype `{lv1['input_dtype']}`"
    )
    c2.markdown(
        f"**Output shape**  \n"
        f"`{lv1['output_shape']}`"
    )

    if lv1["parameters"]:
        st.markdown("**Parameters**")
        st.json(lv1["parameters"])

    if lv1["validation"]:
        st.markdown("**Validation**")
        st.json(lv1["validation"])


# ---------------------------------------------------------------------------
# Level 2 (with step navigation)
# ---------------------------------------------------------------------------
def _render_level2(
    trace: dict[str, Any],
    key_prefix: str,
) -> None:
    steps = build_level2(trace)

    if not steps:
        st.info(
            "This operation didn't record a step-by-step derivation. "
            "Level 1 (formula + parameters) and Level 3 (deep inspection, "
            "if available) still apply."
        )
        return

    ctrl = _get_controller(key_prefix, steps)
    total = ctrl.total
    idx = ctrl.current_index

    # Navigation row
    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 3])
    if nav1.button("◀ Prev", key=f"{key_prefix}_prev",
                   disabled=ctrl.is_first):
        ctrl.prev()
        idx = ctrl.current_index
    if nav2.button("Next ▶", key=f"{key_prefix}_next",
                   disabled=ctrl.is_last):
        ctrl.next()
        idx = ctrl.current_index
    if nav3.button("Reset", key=f"{key_prefix}_reset"):
        ctrl.reset()
        idx = ctrl.current_index
    nav4.progress(
        max(0.0, ctrl.progress),
        text=f"Step {max(1, idx + 1)} of {total}",
    )

    # Step body
    step = ctrl.current()
    if step is None:
        st.warning("No step is currently active.")
        return

    st.markdown(f"### Step {step.get('step', '?')} — {step.get('title', '')}")

    expr = step.get("expression")
    if expr:
        st.code(expr, language="text")

    if step.get("latex"):
        st.latex(step["latex"])

    if step.get("values"):
        st.markdown("**Numerical values**")
        st.json(step["values"])


# ---------------------------------------------------------------------------
# Level 3
# ---------------------------------------------------------------------------
def _render_level3(trace: dict[str, Any]) -> None:
    lv3 = build_level3(trace)

    preview = lv3.get("output_preview")
    if preview is not None:
        st.markdown("**Output preview**")
        st.json(preview)

    pt = lv3.get("pixel_trace")
    if pt is not None:
        st.markdown("**Pixel-level trace**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Kernel**")
            st.dataframe(
                np.array(pt["kernel_values"]),
                use_container_width=False,
            )
        with c2:
            st.markdown("**Neighborhood**")
            st.dataframe(
                np.array(pt["neighborhood"]),
                use_container_width=False,
            )

        st.markdown(
            f"**Sum of products = {pt['sum']:.6f}**  \n"
            f"**Result = {pt['result']:.6f}**"
        )

        with st.expander(
            f"All {len(pt['multiplications'])} multiplications",
            expanded=False,
        ):
            for m in pt["multiplications"]:
                st.text(
                    f"({m['u']},{m['v']}): "
                    f"{m['image_value']:.0f} × {m['kernel_value']:.0f} "
                    f"= {m['product']:.2f}"
                )
    else:
        st.caption(
            "No pixel-level trace was recorded for this operation. "
            "Level 3 is only available for convolution-family operations."
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render(
    trace: dict[str, Any],
    key_prefix: str = "mw",
) -> None:
    """Render the three-level workspace from a Trace dict.

    Parameters
    ----------
    trace : dict
        Output of `Trace.to_dict()`. Never recompute — use the trace.
    key_prefix : str
        Must be unique per page to avoid Streamlit widget-key clashes.
    """
    if not isinstance(trace, dict) or "operation" not in trace:
        st.error("render() received an invalid trace dict.")
        return

    has_l3 = has_level3(trace)
    tab_labels = [
        "📐 Level 1 — Formula",
        "🧾 Level 2 — Steps",
    ]
    if has_l3:
        tab_labels.append("🔬 Level 3 — Deep")

    tabs = st.tabs(tab_labels)
    with tabs[0]:
        _render_level1(trace)
    with tabs[1]:
        _render_level2(trace, key_prefix)
    if has_l3:
        with tabs[2]:
            _render_level3(trace)