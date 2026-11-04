"""Tests for mathematical_engine.level_renderer."""

from __future__ import annotations

from mathematical_engine.level_renderer import (
    build_level1,
    build_level2,
    build_level3,
    has_level3,
)


def _sample_trace(include_pixel_trace: bool = False) -> dict:
    intermediates = []
    if include_pixel_trace:
        intermediates.append({
            "kernel_values": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "neighborhood": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "multiplications": [
                {"u": 0, "v": 0, "image_value": 1, "kernel_value": 1,
                 "product": 1}
            ],
            "sum": 45.0,
            "result": 45.0,
        })
    return {
        "operation_id": "test_0001",
        "operation": "mean",
        "formula": "mu = (1/N) sum(x_i)",
        "formula_plain": "mu",
        "description": "Mean",
        "input_shape": [10, 10],
        "input_dtype": "uint8",
        "parameters": {"mode": "test"},
        "derivation": [
            {"step": 1, "title": "Definition", "expression": "x"},
        ],
        "intermediate_results": intermediates,
        "output_shape": [1],
        "output_preview": {"shape": [1], "first_values": [1.0]},
        "execution_time_ms": 1.5,
        "validation": {"ok": True, "warnings": [], "errors": []},
        "timestamp": "2024-01-01T00:00:00Z",
    }


def test_build_level1():
    lv1 = build_level1(_sample_trace())
    assert lv1["operation"] == "mean"
    assert lv1["input_shape"] == [10, 10]
    assert lv1["parameters"] == {"mode": "test"}
    assert lv1["execution_time_ms"] == 1.5


def test_build_level2():
    steps = build_level2(_sample_trace())
    assert len(steps) == 1
    assert steps[0]["title"] == "Definition"


def test_build_level2_empty():
    trace = _sample_trace()
    trace["derivation"] = []
    assert build_level2(trace) == []


def test_has_level3_false():
    assert not has_level3(_sample_trace())


def test_has_level3_true_with_pixel_trace():
    assert has_level3(_sample_trace(include_pixel_trace=True))


def test_build_level3_empty():
    lv3 = build_level3(_sample_trace())
    assert lv3["pixel_trace"] is None
    assert lv3["output_preview"] is not None


def test_build_level3_with_pixel_trace():
    lv3 = build_level3(_sample_trace(include_pixel_trace=True))
    assert lv3["pixel_trace"] is not None
    assert lv3["pixel_trace"]["sum"] == 45.0