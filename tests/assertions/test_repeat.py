"""Tests for ``@mlflow.test`` decorator."""

from __future__ import annotations

import mlflow
from mlflow._assertions.decorator import MLFLOW_TEST_ATTR


def test_bare_marker():
    @mlflow.test
    def bare():
        pass

    assert getattr(bare, MLFLOW_TEST_ATTR) is True


def test_called_marker():
    @mlflow.test()
    def called():
        pass

    assert getattr(called, MLFLOW_TEST_ATTR) is True
