"""Trace manager — records the actual computation of each operation.

Design
------
Each operation wrapped by TraceManager produces a structured dict with
ONLY actual values. Never fabricate.

Trace schema
------------
{
    "operation_id":        str,          # e.g. "transpose_0001"
    "operation":           str,          # e.g. "transpose"
    "formula":             str,          # LaTeX
    "formula_plain":       str,
    "description":         str,
    "input_shape":         list[int],
    "input_dtype":         str,
    "parameters":          dict,
    "derivation":          list[dict],   # populated by explanation_templates
    "intermediate_results": list[dict],
    "output_shape":        list[int] | None,
    "output_preview":      list | None,  # small ndarray preview
    "execution_time_ms":   float,
    "validation":          dict,
    "timestamp":           str,
}

Session
-------
TraceManager maintains an in-memory list of traces for the current session.
It is created once per Streamlit session (in app.py) and stored in
st.session_state. It never imports Streamlit — the UI passes it around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

import numpy as np

from mathematical_engine.formula_registry import get_formula


# ---------------------------------------------------------------------------
# Dataclass wrapper (optional but convenient)
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    """Lightweight container for a single trace record."""

    operation_id: str
    operation: str
    formula: str
    formula_plain: str
    description: str
    input_shape: list[int]
    input_dtype: str
    parameters: dict[str, Any] = field(default_factory=dict)
    derivation: list[dict[str, Any]] = field(default_factory=list)
    intermediate_results: list[dict[str, Any]] = field(default_factory=list)
    output_shape: list[int] | None = None
    output_preview: list | None = None
    execution_time_ms: float = 0.0
    validation: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation": self.operation,
            "formula": self.formula,
            "formula_plain": self.formula_plain,
            "description": self.description,
            "input_shape": self.input_shape,
            "input_dtype": self.input_dtype,
            "parameters": dict(self.parameters),
            "derivation": list(self.derivation),
            "intermediate_results": list(self.intermediate_results),
            "output_shape": self.output_shape,
            "output_preview": self.output_preview,
            "execution_time_ms": self.execution_time_ms,
            "validation": dict(self.validation),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------
def _preview(obj: Any, max_cells: int = 36) -> Any:
    """Return a small, JSON-safe preview of arrays / dicts / scalars.

    Keeps the trace lightweight even for large images.
    """
    if isinstance(obj, np.ndarray):
        flat = obj.ravel()
        n = min(max_cells, flat.size)
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "first_values": [float(v) for v in flat[:n]],
        }
    if isinstance(obj, (list, tuple)):
        return [_preview(v, max_cells) for v in obj[:max_cells]]
    if isinstance(obj, dict):
        return {k: _preview(v, max_cells) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# TraceManager
# ---------------------------------------------------------------------------
class TraceManager:
    """Records every operation performed in the current session."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def next_id(self, operation: str) -> str:
        """Return a unique operation id like 'transpose_0003'."""
        self._counter += 1
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in operation)
        return f"{safe}_{self._counter:04d}"

    def record(
        self,
        operation: str,
        input_array: np.ndarray,
        parameters: dict[str, Any] | None = None,
        derivation: list[dict[str, Any]] | None = None,
        intermediate_results: list[dict[str, Any]] | None = None,
        output: Any = None,
        validation: dict[str, Any] | None = None,
        execution_time_ms: float = 0.0,
    ) -> Trace:
        """Record a fully-formed trace. Returns the stored Trace."""
        formula = get_formula(operation)
        trace = Trace(
            operation_id=self.next_id(operation),
            operation=operation,
            formula=formula["latex"],
            formula_plain=formula["plain"],
            description=formula["description"],
            input_shape=list(input_array.shape),
            input_dtype=str(input_array.dtype),
            parameters=dict(parameters or {}),
            derivation=list(derivation or []),
            intermediate_results=list(intermediate_results or []),
            output_shape=(
                list(output.shape) if isinstance(output, np.ndarray) else None
            ),
            output_preview=_preview(output),
            execution_time_ms=float(execution_time_ms),
            validation=dict(validation or {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.traces.append(trace)
        return trace

    def wrap(
        self,
        operation: str,
        fn: Callable[..., Any],
        *args: Any,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Trace]:
        """Execute fn(*args, **kwargs), record it, and return (output, trace).

        Timing is measured around the call.
        """
        if not args:
            raise ValueError("wrap() requires at least one array argument.")

        input_array = args[0]
        if not isinstance(input_array, np.ndarray):
            raise TypeError(
                "First argument to wrap() must be a NumPy array."
            )

        t0 = perf_counter()
        output = fn(*args, **kwargs)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        trace = self.record(
            operation=operation,
            input_array=input_array,
            parameters=parameters,
            output=output,
            execution_time_ms=elapsed_ms,
        )
        return output, trace

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def get(self, operation_id: str) -> Trace | None:
        """Look up a trace by its id."""
        for t in self.traces:
            if t.operation_id == operation_id:
                return t
        return None

    def last(self) -> Trace | None:
        """Return the most recent trace, or None."""
        return self.traces[-1] if self.traces else None

    def clear(self) -> None:
        self.traces.clear()
        self._counter = 0

    def as_dicts(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.traces]