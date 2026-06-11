"""Tests for the @mlflow.test pytest plugin."""

from __future__ import annotations

import mlflow
from mlflow._assertions import session as _session
from mlflow._assertions.decorator import MLFLOW_TEST_ATTR

# ---------------------------------------------------------------------------
# @mlflow.test decorator
# ---------------------------------------------------------------------------


def test_bare_marker():
    @mlflow.test
    def f():
        pass

    assert getattr(f, MLFLOW_TEST_ATTR) is True


def test_called_marker():
    @mlflow.test()
    def f():
        pass

    assert getattr(f, MLFLOW_TEST_ATTR) is True


# ---------------------------------------------------------------------------
# Plugin sets current test name for @mlflow.test-marked tests
# ---------------------------------------------------------------------------


@mlflow.test
def test_current_test_is_set_inside_mlflow_test():
    name, case_id = _session.current_test()
    assert name == "test_current_test_is_set_inside_mlflow_test"
    assert case_id is None


def test_current_test_is_none_outside_mlflow_test():
    name, _ = _session.current_test()
    assert name is None


# ---------------------------------------------------------------------------
# Plugin creates a run for the session
# ---------------------------------------------------------------------------


@mlflow.test
def test_run_is_created():
    assert mlflow.active_run() is not None
