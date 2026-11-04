"""Tests for mathematical_engine.step_controller."""

from __future__ import annotations

import pytest

from mathematical_engine.step_controller import StepController


STEPS = [
    {"step": 1, "title": "A", "expression": "1"},
    {"step": 2, "title": "B", "expression": "2"},
    {"step": 3, "title": "C", "expression": "3"},
]


def test_init_no_steps():
    c = StepController()
    assert c.total == 0
    assert c.current() is None
    assert c.current_index == -1


def test_init_with_steps():
    c = StepController(STEPS)
    assert c.total == 3
    assert c.current_index == -1        # not yet started
    assert c.current() is None


def test_reset():
    c = StepController(STEPS)
    c.reset()
    assert c.current_index == 0
    assert c.current()["title"] == "A"


def test_next_moves_forward():
    c = StepController(STEPS)
    c.reset()
    c.next()
    assert c.current()["title"] == "B"
    c.next()
    assert c.current()["title"] == "C"


def test_next_stops_at_end():
    c = StepController(STEPS)
    c.reset()
    for _ in range(10):
        c.next()
    assert c.is_last
    assert c.current()["title"] == "C"


def test_prev_moves_back():
    c = StepController(STEPS)
    c.reset()
    c.go_to(2)
    c.prev()
    assert c.current()["title"] == "B"


def test_prev_stops_at_zero():
    c = StepController(STEPS)
    c.reset()
    for _ in range(10):
        c.prev()
    assert c.is_first
    assert c.current_index == 0


def test_go_to_valid():
    c = StepController(STEPS)
    c.go_to(1)
    assert c.current()["title"] == "B"


def test_go_to_out_of_range():
    c = StepController(STEPS)
    with pytest.raises(IndexError):
        c.go_to(10)


def test_progress():
    c = StepController(STEPS)
    c.reset()
    assert c.progress == pytest.approx(1 / 3)
    c.next()
    assert c.progress == pytest.approx(2 / 3)
    c.next()
    assert c.progress == pytest.approx(1.0)


def test_describe():
    c = StepController(STEPS)
    c.reset()
    d = c.describe()
    assert d["total"] == 3
    assert d["current_index"] == 0
    assert d["current_title"] == "A"
    assert d["is_first"]
    assert not d["is_last"]