"""Tests for mathematical_engine (Phase 2 foundation)."""

from __future__ import annotations

import numpy as np
import pytest

from mathematical_engine.computation_validator import (
    require,
    validate_array,
    validate_intensity_range,
    validate_multiply_shapes,
    validate_same_shape,
)
from mathematical_engine.explanation_templates import (
    explain_add,
    explain_gamma,
    explain_matrix_multiply,
    explain_mean,
    explain_negative,
    explain_transpose,
)
from mathematical_engine.formula_registry import FORMULAS, get_formula
from mathematical_engine.matrix_inspector import (
    block_preview,
    center_pixel,
    column_preview,
    neighborhood,
    row_preview,
)
from mathematical_engine.numerical_calculator import (
    difference_norm,
    maximum_abs_difference,
    relative_error,
    safe_stats,
)
from mathematical_engine.trace_manager import TraceManager


# ---------------------------------------------------------------------------
# formula_registry
# ---------------------------------------------------------------------------
def test_formula_registry_has_core_entries():
    for key in [
        "transpose", "add", "subtract", "scalar_multiply",
        "elementwise_multiply", "matrix_multiply",
        "mean", "variance", "std", "frobenius",
        "identity", "linear", "negative", "log", "gamma", "contrast_stretch",
    ]:
        assert key in FORMULAS


def test_get_formula_raises_on_unknown():
    with pytest.raises(KeyError):
        get_formula("not_a_real_operation")


# ---------------------------------------------------------------------------
# trace_manager
# ---------------------------------------------------------------------------
def test_trace_manager_records_and_returns(sample_gray):
    tm = TraceManager()
    A = sample_gray[:8, :8]
    out = A.T
    tr = tm.record(
        operation="transpose",
        input_array=A,
        parameters={"method": "A.T"},
        output=out,
        execution_time_ms=1.23,
        validation={"ok": True, "warnings": [], "errors": []},
    )
    assert tr.operation == "transpose"
    assert tr.input_shape == [8, 8]
    assert tr.output_shape == [8, 8]
    assert tr.execution_time_ms == 1.23
    assert tm.last() is tr
    assert len(tm.traces) == 1
    assert tr.operation_id.startswith("transpose_")


def test_trace_manager_wrap(sample_gray):
    tm = TraceManager()
    A = sample_gray[:4, :4]
    out, trace = tm.wrap(
        "transpose",
        lambda x: x.T,
        A,
        parameters={"method": "A.T"},
    )
    assert np.array_equal(out, A.T)
    assert trace.operation == "transpose"
    assert trace.execution_time_ms >= 0.0


def test_trace_manager_clear(sample_gray):
    tm = TraceManager()
    tm.record("transpose", sample_gray[:4, :4], output=sample_gray[:4, :4].T)
    assert len(tm.traces) == 1
    tm.clear()
    assert len(tm.traces) == 0
    assert tm.last() is None


# ---------------------------------------------------------------------------
# matrix_inspector
# ---------------------------------------------------------------------------
def test_neighborhood_center(sample_gray):
    nb = neighborhood(sample_gray, 10, 10, size=3)
    assert nb.shape == (3, 3)
    assert nb[1, 1] == sample_gray[10, 10]


def test_neighborhood_clips_at_border(sample_gray):
    nb = neighborhood(sample_gray, 0, 0, size=5)
    assert nb.shape == (3, 3)


def test_neighborhood_rejects_even_size(sample_gray):
    with pytest.raises(ValueError):
        neighborhood(sample_gray, 5, 5, size=4)


def test_row_column_preview(sample_gray):
    rp = row_preview(sample_gray, 0, max_values=5)
    cp = column_preview(sample_gray, 0, max_values=5)
    assert rp["length"] == 64
    assert cp["length"] == 64
    assert len(rp["first_values"]) == 5


def test_block_preview(sample_gray):
    bp = block_preview(sample_gray, 0, 0, 3, 3)
    assert bp["shape"] == [3, 3]
    assert len(bp["values"]) == 3


def test_center_pixel(sample_gray):
    r, c = center_pixel(sample_gray)
    assert (r, c) == (32, 32)


# ---------------------------------------------------------------------------
# numerical_calculator
# ---------------------------------------------------------------------------
def test_safe_stats(sample_gray):
    s = safe_stats(sample_gray)
    assert s["size"] == 64 * 64
    assert s["mean"] == pytest.approx(float(sample_gray.mean()))


def test_safe_stats_empty():
    s = safe_stats(np.array([], dtype=np.float64))
    assert s["size"] == 0
    assert s["mean"] is None


def test_maximum_abs_difference():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 6]])
    assert maximum_abs_difference(A, B) == 2.0


def test_difference_norm():
    A = np.zeros((2, 2))
    B = np.array([[3, 0], [0, 4]])
    assert difference_norm(A, B) == pytest.approx(5.0)


def test_relative_error_zero_denominator():
    A = np.ones((2, 2))
    B = np.zeros((2, 2))
    assert relative_error(A, B) is None


# ---------------------------------------------------------------------------
# computation_validator
# ---------------------------------------------------------------------------
def test_validate_array_ok(sample_gray):
    r = validate_array(sample_gray)
    assert r["ok"] is True


def test_validate_array_not_numpy():
    r = validate_array([[1, 2], [3, 4]])
    assert r["ok"] is False


def test_validate_array_wrong_dim():
    r = validate_array(np.zeros((2, 2, 3)))
    assert r["ok"] is False


def test_validate_same_shape_ok():
    r = validate_same_shape(np.zeros((2, 2)), np.ones((2, 2)))
    assert r["ok"] is True


def test_validate_same_shape_mismatch():
    r = validate_same_shape(np.zeros((2, 2)), np.ones((3, 3)))
    assert r["ok"] is False


def test_validate_multiply_shapes_ok():
    r = validate_multiply_shapes(np.zeros((2, 3)), np.zeros((3, 4)))
    assert r["ok"] is True


def test_validate_multiply_shapes_mismatch():
    r = validate_multiply_shapes(np.zeros((2, 3)), np.zeros((4, 5)))
    assert r["ok"] is False


def test_validate_intensity_range_warns():
    A = np.array([[-1, 300]], dtype=np.int32)
    r = validate_intensity_range(A)
    assert len(r["warnings"]) == 1


def test_require_raises_on_error():
    with pytest.raises(ValueError):
        require({"ok": False, "errors": ["bad"], "warnings": []})


# ---------------------------------------------------------------------------
# explanation_templates
# ---------------------------------------------------------------------------
def test_explain_transpose_returns_steps():
    A = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    steps = explain_transpose(A, A.T)
    assert len(steps) >= 3
    assert all("step" in s and "title" in s for s in steps)


def test_explain_add_matches_values():
    A = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    B = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    C = A + B
    steps = explain_add(A, B, C)
    spot = [s for s in steps if "Spot check" in s["title"]][0]
    assert spot["values"]["C_00"] == int(C[0, 0])


def test_explain_matrix_multiply_has_output_shape():
    A = np.ones((2, 3))
    B = np.ones((3, 4))
    C = A @ B
    steps = explain_matrix_multiply(A, B, C)
    shape_step = [s for s in steps if s["title"] == "Output shape"][0]
    assert shape_step["values"]["shape"] == [2, 4]


def test_explain_mean_uses_actual_array():
    A = np.array([[2, 4], [6, 8]], dtype=np.float64)
    steps = explain_mean(A)
    compute = [s for s in steps if s["title"] == "Compute"][0]
    assert compute["values"]["mean"] == pytest.approx(A.mean())


def test_explain_gamma_spot_check():
    A = np.array([[128]], dtype=np.uint8)
    C = np.array([[255.0 * (128.0 / 255.0) ** 2.0]])
    steps = explain_gamma(A, 2.0, C)
    spot = [s for s in steps if "Spot check" in s["title"]][0]
    assert spot["values"]["clipped"] == pytest.approx(float(C[0, 0]))


def test_explain_negative():
    A = np.array([[10, 200]], dtype=np.uint8)
    C = 255 - A
    steps = explain_negative(A, C)
    assert len(steps) == 2