"""Reusable side-by-side image comparison widgets."""

from __future__ import annotations

import numpy as np
import streamlit as st


def side_by_side(
    original: np.ndarray,
    processed: np.ndarray,
    labels: tuple[str, str] = ("Original", "Processed"),
    clamp: bool = True,
) -> None:
    """Two-column image comparison with matching captions."""
    if original.shape != processed.shape:
        st.warning(
            f"Shape mismatch: {original.shape} vs {processed.shape}. "
            "Showing each independently."
        )
    c1, c2 = st.columns(2)
    c1.image(original, caption=labels[0], use_column_width=True,
             clamp=clamp)
    c2.image(processed, caption=labels[1], use_column_width=True,
             clamp=clamp)


def difference_view(
    original: np.ndarray,
    processed: np.ndarray,
    label: str = "|Processed − Original|",
) -> None:
    """Show the absolute difference as an amplified heatmap."""
    if original.shape != processed.shape:
        st.warning("Shape mismatch — difference not available.")
        return
    diff = np.abs(
        original.astype(np.float64) - processed.astype(np.float64)
    )
    dmax = float(diff.max())
    st.caption(f"{label} · max = {dmax:.2f}")
    if dmax > 0:
        vis = (diff / dmax * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(processed, dtype=np.uint8)
    st.image(vis, caption=label, use_column_width=True, clamp=True)


def metric_cards(metrics: dict[str, float | int | str]) -> None:
    """Render a row of metric cards."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, v)