"""Tests for mathematical_engine.history."""

from __future__ import annotations

import numpy as np
import pytest

from mathematical_engine.history import (
    filtered_by_operation,
    recent,
    summarize,
    to_rows,
)
from mathematical_engine.trace_manager import TraceManager


def _make_traces(n: int = 5) -> list:
    tm = TraceManager()
    A = np.zeros((4, 4), dtype=np.uint8)
    for i in range(n):
        op = "transpose" if i % 2 == 0 else "mean"
        tm.record(operation=op, input_array=A, output=A,
                  execution_time_ms=float(i + 1))
    return tm.traces


def test_recent():
    traces = _make_traces(5)
    r = recent(traces, n=3)
    assert len(r) == 3
    # Newest first
    assert r[0] is traces[-1]


def test_recent_zero():
    traces = _make_traces(3)
    assert recent(traces, n=0) == []


def test_recent_invalid():
    with pytest.raises(ValueError):
        recent(_make_traces(2), n=-1)


def test_filtered_by_operation():
    traces = _make_traces(5)
    only_transpose = filtered_by_operation(traces, ["transpose"])
    assert len(only_transpose) == 3
    assert all(t.operation == "transpose" for t in only_transpose)


def test_summarize_empty():
    s = summarize([])
    assert s["count"] == 0
    assert s["last_operation"] is None


def test_summarize_non_empty():
    traces = _make_traces(4)
    s = summarize(traces)
    assert s["count"] == 4
    assert s["operations"]["transpose"] == 2
    assert s["operations"]["mean"] == 2
    assert s["last_operation"] == traces[-1].operation
    assert s["total_time_ms"] == pytest.approx(1 + 2 + 3 + 4)


def test_to_rows():
    traces = _make_traces(3)
    rows = to_rows(traces)
    assert len(rows) == 3
    assert set(rows[0].keys()) == {
        "operation_id", "operation", "input_shape", "output_shape",
        "input_dtype", "time_ms", "parameters", "timestamp",
    }
    assert rows[0]["input_shape"] == "4x4"