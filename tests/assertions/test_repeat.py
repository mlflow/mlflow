"""Tests for ``@mlflow.test`` decorator validation."""

from __future__ import annotations

import pytest

import mlflow
from mlflow._assertions.decorator import (
    MLFLOW_TEST_PASS_THRESHOLD_ATTR,
    MLFLOW_TEST_REPEAT_ATTR,
)


def test_default_threshold_is_strict_majority():
    @mlflow.test(repeat=3)
    def three():
        pass

    @mlflow.test(repeat=5)
    def five():
        pass

    assert getattr(three, MLFLOW_TEST_REPEAT_ATTR) == 3
    assert getattr(three, MLFLOW_TEST_PASS_THRESHOLD_ATTR) == 2
    assert getattr(five, MLFLOW_TEST_PASS_THRESHOLD_ATTR) == 3


def test_bare_marker_is_single_shot():
    @mlflow.test
    def bare():
        pass

    @mlflow.test()
    def called():
        pass

    for fn in (bare, called):
        assert getattr(fn, MLFLOW_TEST_REPEAT_ATTR) == 1
        assert getattr(fn, MLFLOW_TEST_PASS_THRESHOLD_ATTR) == 1


@pytest.mark.parametrize("repeat", [0, -1])
def test_invalid_repeat_raises(repeat):
    with pytest.raises(ValueError, match="must be >= 1"):

        @mlflow.test(repeat=repeat)
        def t():
            pass


@pytest.mark.parametrize("threshold", [0, 4])
def test_threshold_out_of_range_raises(threshold):
    with pytest.raises(ValueError, match="between 1 and repeat"):

        @mlflow.test(repeat=3, pass_threshold=threshold)
        def t():
            pass
