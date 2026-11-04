"""Tests for core.reference_dataset."""

from __future__ import annotations

import numpy as np
import pytest

from core.reference_dataset import (
    describe_stack,
    preprocess_stack,
    synthesize_reference_stack,
)


def test_synthesize_shape_and_dtype():
    stack = synthesize_reference_stack(n_images=5, size=32, seed=1)
    assert stack.shape == (5, 32, 32)
    assert stack.dtype == np.uint8


def test_synthesize_deterministic():
    a = synthesize_reference_stack(n_images=3, size=16, seed=42)
    b = synthesize_reference_stack(n_images=3, size=16, seed=42)
    assert np.array_equal(a, b)


def test_synthesize_rejects_too_few_images():
    with pytest.raises(ValueError):
        synthesize_reference_stack(n_images=1, size=16)


def test_synthesize_rejects_too_small_size():
    with pytest.raises(ValueError):
        synthesize_reference_stack(n_images=2, size=8)


def test_preprocess_stack_no_resize():
    stack = np.zeros((3, 10, 10), dtype=np.uint8)
    out = preprocess_stack(stack, target_size=None)
    assert out.shape == (3, 10, 10)


def test_preprocess_stack_resize():
    stack = np.zeros((3, 20, 20), dtype=np.uint8)
    out = preprocess_stack(stack, target_size=10)
    assert out.shape == (3, 10, 10)


def test_describe_stack():
    stack = np.zeros((4, 16, 16), dtype=np.uint8)
    info = describe_stack(stack)
    assert info["n_images"] == 4
    assert info["height"] == 16
    assert info["width"] == 16
    assert info["pixel_count_per_image"] == 256