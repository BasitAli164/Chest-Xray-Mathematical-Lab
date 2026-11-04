"""Processing history — cross-page ordered list of traces.

TraceManager already stores a list. This module adds convenience
queries for the dashboard and sidebar: filter by operation, get recent,
get statistics, export as CSV.

Public API
----------
- recent(traces, n=10)
- filtered_by_operation(traces, ops)
- summarize(traces)
- to_rows(traces)
"""

from __future__ import annotations

from typing import Any, Iterable

from mathematical_engine.trace_manager import Trace


def recent(traces: list[Trace], n: int = 10) -> list[Trace]:
    """Return the N most recent traces (most recent first)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")
    if n == 0:
        return []
    return list(reversed(traces[-n:]))

def filtered_by_operation(
    traces: list[Trace],
    ops: Iterable[str],
) -> list[Trace]:
    """Return traces whose operation is in `ops`, newest first."""
    ops_set = set(ops)
    return [t for t in reversed(traces) if t.operation in ops_set]


def summarize(traces: list[Trace]) -> dict[str, Any]:
    """Aggregate counts, total time, and last operation."""
    if not traces:
        return {
            "count": 0,
            "total_time_ms": 0.0,
            "mean_time_ms": 0.0,
            "operations": {},
            "last_operation": None,
        }

    ops: dict[str, int] = {}
    total_time = 0.0
    for t in traces:
        ops[t.operation] = ops.get(t.operation, 0) + 1
        total_time += float(t.execution_time_ms)

    return {
        "count": len(traces),
        "total_time_ms": total_time,
        "mean_time_ms": total_time / len(traces),
        "operations": ops,
        "last_operation": traces[-1].operation,
    }


def to_rows(traces: list[Trace]) -> list[dict[str, Any]]:
    """Convert traces to flat dicts suitable for st.dataframe / CSV."""
    rows: list[dict[str, Any]] = []
    for t in traces:
        rows.append({
            "operation_id": t.operation_id,
            "operation": t.operation,
            "input_shape": "x".join(str(s) for s in t.input_shape),
            "output_shape": (
                "x".join(str(s) for s in t.output_shape)
                if t.output_shape else ""
            ),
            "input_dtype": t.input_dtype,
            "time_ms": round(float(t.execution_time_ms), 3),
            "parameters": str(t.parameters),
            "timestamp": t.timestamp,
        })
    return rows