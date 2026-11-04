"""Three-level transparency renderer — pure logic layer.

The actual UI calls st.* — this module *prepares* the three views from a
Trace dict so the UI layer stays thin and testable.

Levels
------
Level 1 — overview : formula, dimensions, parameters, short summary
Level 2 — steps    : numbered derivation steps (from trace["derivation"])
Level 3 — deep     : matrix preview, pixel-level trace (if present in
                     trace["intermediate_results"])

Public API
----------
- build_level1(trace)
- build_level2(trace)
- build_level3(trace)
- has_level3(trace)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Level 1 — formula & overview
# ---------------------------------------------------------------------------
def build_level1(trace: dict[str, Any]) -> dict[str, Any]:
    """Compose the overview dict from an already-recorded Trace.

    All values come directly from the trace — nothing is recomputed.
    """
    return {
        "operation_id": trace.get("operation_id", "?"),
        "operation": trace.get("operation", "?"),
        "formula_latex": trace.get("formula", ""),
        "formula_plain": trace.get("formula_plain", ""),
        "description": trace.get("description", ""),
        "input_shape": trace.get("input_shape"),
        "input_dtype": trace.get("input_dtype"),
        "output_shape": trace.get("output_shape"),
        "parameters": trace.get("parameters", {}),
        "execution_time_ms": float(trace.get("execution_time_ms", 0.0)),
        "validation": trace.get("validation", {}),
        "timestamp": trace.get("timestamp", ""),
    }


# ---------------------------------------------------------------------------
# Level 2 — step-by-step derivation
# ---------------------------------------------------------------------------
def build_level2(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of derivation steps from the trace.

    Each step must have at least `step`, `title`, `expression`. `latex`
    and `values` are optional.
    """
    derivation = trace.get("derivation") or []
    # Defensive copy so the caller can't mutate the trace through here.
    return [dict(step) for step in derivation]


# ---------------------------------------------------------------------------
# Level 3 — deep numerical inspection
# ---------------------------------------------------------------------------
def has_level3(trace: dict[str, Any]) -> bool:
    """A trace has Level 3 data **only** if it contains a pixel-level trace
    (i.e. a dict with 'multiplications' inside intermediate_results).

    A simple output_preview with 'shape' is NOT enough — Level 3 is meant
    for deep numerical inspection (individual multiplications, local
    neighborhood, intermediate sums). Those only come from
    convolution-family operations.
    """
    intermediates = trace.get("intermediate_results") or []
    for item in intermediates:
        if isinstance(item, dict) and "multiplications" in item:
            return True
    return False


def build_level3(trace: dict[str, Any]) -> dict[str, Any]:
    """Compose the deep-inspection dict from the trace.

    Includes whatever is available:
      - output_preview  (small array summary, always present if any output)
      - pixel_trace     (if the trace recorded one in intermediate_results)
    """
    intermediates = trace.get("intermediate_results") or []
    pixel_trace = None
    for item in intermediates:
        if isinstance(item, dict) and "multiplications" in item:
            pixel_trace = item
            break

    return {
        "output_preview": trace.get("output_preview"),
        "pixel_trace": pixel_trace,
        "parameters": trace.get("parameters", {}),
        "input_shape": trace.get("input_shape"),
        "output_shape": trace.get("output_shape"),
    }