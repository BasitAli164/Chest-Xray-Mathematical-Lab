"""Step-by-step navigation controller.

Pure-logic module (no Streamlit imports) so it is fully unit-testable.

A StepController tracks a cursor over a list of derivation steps. It is
created once per operation, persisted in st.session_state, and used by
the UI to move forward/backward.

Public API
----------
- StepController(steps: list[dict])
    .current_index -> int
    .current() -> dict | None
    .next() -> dict | None
    .prev() -> dict | None
    .go_to(i) -> dict | None
    .reset()
    .is_first / .is_last
    .total -> int
    .progress -> float
"""

from __future__ import annotations

from typing import Any


class StepController:
    """Cursor-based controller for step-by-step derivation navigation.

    Parameters
    ----------
    steps : list[dict]
        A list of step dictionaries as produced by
        `explanation_templates.*` builders. Each should have at least
        `step`, `title`, `expression`.

    Attributes
    ----------
    _steps : list[dict]
    _index : int   0-based cursor; -1 means "not started"
    """

    def __init__(self, steps: list[dict[str, Any]] | None = None) -> None:
        self._steps: list[dict[str, Any]] = list(steps or [])
        self._index: int = -1 if self._steps else -1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self._steps)

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def is_first(self) -> bool:
        return self._index <= 0

    @property
    def is_last(self) -> bool:
        return self._index >= self.total - 1

    @property
    def progress(self) -> float:
        """Fraction of steps completed, in [0.0, 1.0]."""
        if self.total == 0:
            return 0.0
        return (self._index + 1) / self.total

    # ------------------------------------------------------------------
    # Mutators / accessors
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Start over at step 0 (or -1 if empty)."""
        self._index = 0 if self._steps else -1

    def current(self) -> dict[str, Any] | None:
        if not self._steps or self._index < 0 or self._index >= self.total:
            return None
        return self._steps[self._index]

    def next(self) -> dict[str, Any] | None:
        if not self._steps:
            return None
        self._index = min(self._index + 1, self.total - 1)
        return self.current()

    def prev(self) -> dict[str, Any] | None:
        if not self._steps:
            return None
        self._index = max(self._index - 1, 0)
        return self.current()

    def go_to(self, i: int) -> dict[str, Any] | None:
        if not self._steps:
            return None
        if not (0 <= i < self.total):
            raise IndexError(
                f"Step index {i} out of range [0, {self.total - 1}]."
            )
        self._index = i
        return self.current()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe summary (used by tests and diagnostics)."""
        return {
            "total": self.total,
            "current_index": self._index,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "progress": self.progress,
            "current_title": (self.current() or {}).get("title"),
        }