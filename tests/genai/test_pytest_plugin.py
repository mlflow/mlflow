"""Tests for the MLflow GenAI pytest plugin."""

from __future__ import annotations

from unittest import mock

import pytest

import mlflow
from mlflow.genai.pytest_plugin import (
    MLFLOW_RUN_TYPE_PYTEST,
    MLFLOW_RUN_TYPE_TAG,
    MLFLOW_TEST_DURATION_TAG,
    MLFLOW_TEST_OUTCOME_TAG,
    TestResult,
)

# Enable pytester fixture
pytest_plugins = ["pytester"]


@pytest.fixture
def _isolated_tracking(tmp_path):
    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
    mlflow.set_tracking_uri(None)


# ---------------------------------------------------------------------------
# Module export tests
# ---------------------------------------------------------------------------


def test_plugin_module_exports():
    from mlflow.genai import pytest_plugin

    assert hasattr(pytest_plugin, "mlflow_run")
    assert hasattr(pytest_plugin, "mlflow_experiment_name")
    assert hasattr(pytest_plugin, "mlflow_evaluate")
    assert hasattr(pytest_plugin, "pytest_addoption")
    assert hasattr(pytest_plugin, "pytest_configure")
    assert hasattr(pytest_plugin, "pytest_collection_modifyitems")
    assert hasattr(pytest_plugin, "pytest_runtest_call")
    assert hasattr(pytest_plugin, "pytest_runtest_makereport")
    assert hasattr(pytest_plugin, "pytest_terminal_summary")


def test_plugin_has_pytest11_entry_point():
    from importlib.metadata import entry_points

    eps = entry_points(group="pytest11")
    mlflow_eps = [ep for ep in eps if ep.name == "mlflow-genai"]
    assert len(mlflow_eps) == 1
    assert mlflow_eps[0].value == "mlflow.genai.pytest_plugin"


# ---------------------------------------------------------------------------
# Nested run test
# ---------------------------------------------------------------------------


def test_nested_run_has_parent_id(_isolated_tracking):
    mlflow.set_experiment("pytest")
    with mlflow.start_run(run_name="parent") as parent_run:
        parent_id = parent_run.info.run_id
        with mlflow.start_run(run_name="child_test", nested=True) as child_run:
            assert child_run.info.run_id != parent_id
            client = mlflow.MlflowClient()
            child_info = client.get_run(child_run.info.run_id)
            assert child_info.data.tags.get("mlflow.parentRunId") == parent_id


# ---------------------------------------------------------------------------
# TestResult dataclass
# ---------------------------------------------------------------------------


def test_test_result_defaults():
    r = TestResult(name="test_foo", outcome="passed", duration_s=1.23)
    assert r.name == "test_foo"
    assert r.outcome == "passed"
    assert r.duration_s == 1.23
    assert r.metrics == {}
    assert r.run_id is None


def test_test_result_with_metrics():
    r = TestResult(
        name="test_bar",
        outcome="failed",
        duration_s=0.5,
        metrics={"accuracy": 0.9},
        run_id="abc123",
    )
    assert r.metrics == {"accuracy": 0.9}
    assert r.run_id == "abc123"


def test_test_result_is_frozen():
    r = TestResult(name="test_baz", outcome="passed", duration_s=0.1)
    with pytest.raises(AttributeError):
        r.name = "other"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    assert MLFLOW_RUN_TYPE_TAG == "mlflow.runType"
    assert MLFLOW_RUN_TYPE_PYTEST == "pytest"
    assert MLFLOW_TEST_OUTCOME_TAG == "mlflow.test.outcome"
    assert MLFLOW_TEST_DURATION_TAG == "mlflow.test.duration"


# ---------------------------------------------------------------------------
# pytest_configure
# ---------------------------------------------------------------------------


def test_pytest_configure_sets_attributes():
    from mlflow.genai.pytest_plugin import pytest_configure

    config = mock.MagicMock()
    config.addinivalue_line = mock.MagicMock()
    pytest_configure(config)

    assert config._mlflow_genai_active is False
    assert config._mlflow_genai_results == []
    assert config._mlflow_genai_parent_run_id is None
    assert config._mlflow_genai_experiment_name is None
    config.addinivalue_line.assert_called_once()


# ---------------------------------------------------------------------------
# pytest_collection_modifyitems
# ---------------------------------------------------------------------------


def test_collection_modifyitems_activates_on_genai_marker():
    from mlflow.genai.pytest_plugin import pytest_collection_modifyitems

    config = mock.MagicMock()
    config._mlflow_genai_active = False

    item_with_marker = mock.MagicMock()
    item_with_marker.get_closest_marker.return_value = mock.MagicMock()

    item_without_marker = mock.MagicMock()
    item_without_marker.get_closest_marker.return_value = None

    pytest_collection_modifyitems(config, [item_without_marker, item_with_marker])
    assert config._mlflow_genai_active is True


def test_collection_modifyitems_stays_inactive_without_marker():
    from mlflow.genai.pytest_plugin import pytest_collection_modifyitems

    config = mock.MagicMock()
    config._mlflow_genai_active = False

    item = mock.MagicMock()
    item.get_closest_marker.return_value = None

    pytest_collection_modifyitems(config, [item])
    assert config._mlflow_genai_active is False


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def test_terminal_summary_skips_when_inactive():
    from mlflow.genai.pytest_plugin import pytest_terminal_summary

    config = mock.MagicMock()
    config._mlflow_genai_active = False

    reporter = mock.MagicMock()
    pytest_terminal_summary(reporter, 0, config)
    reporter.section.assert_not_called()


def test_terminal_summary_skips_when_no_results():
    from mlflow.genai.pytest_plugin import pytest_terminal_summary

    config = mock.MagicMock()
    config._mlflow_genai_active = True
    config._mlflow_genai_results = []

    reporter = mock.MagicMock()
    pytest_terminal_summary(reporter, 0, config)
    reporter.section.assert_not_called()


def test_terminal_summary_prints_table():
    from mlflow.genai.pytest_plugin import pytest_terminal_summary

    config = mock.MagicMock()
    config._mlflow_genai_active = True
    config._mlflow_genai_results = [
        TestResult(name="test_a", outcome="passed", duration_s=1.0),
        TestResult(
            name="test_b",
            outcome="failed",
            duration_s=2.5,
            metrics={"score": 0.8},
        ),
    ]
    config._mlflow_genai_experiment_name = "my_exp"
    config._mlflow_genai_parent_run_id = "run-123"
    config._mlflow_auto_started_parent = False

    reporter = mock.MagicMock()
    pytest_terminal_summary(reporter, 0, config)

    reporter.section.assert_called_once_with("MLflow GenAI Test Results")
    lines = [call.args[0] for call in reporter.write_line.call_args_list]
    assert any("1 passed" in line and "1 failed" in line for line in lines)
    assert any("my_exp" in line for line in lines)
    assert any("run-123" in line for line in lines)


# ---------------------------------------------------------------------------
# Integration: run with pytester
# ---------------------------------------------------------------------------


@pytest.fixture
def pytester_with_plugin(pytester, tmp_path):
    db_path = str(tmp_path / "mlflow_test.db").replace("\\", "/")
    pytester.makeconftest(
        f"""
import mlflow
mlflow.set_tracking_uri("sqlite:///{db_path}")
"""
    )
    return pytester, db_path


def test_marker_integration(pytester_with_plugin):
    pytester, db_path = pytester_with_plugin
    pytester.makepyfile(
        """
import pytest

@pytest.mark.genai
def test_example():
    assert True
"""
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*MLflow GenAI Test Results*"])


def test_marker_with_parametrize_integration(pytester_with_plugin):
    pytester, db_path = pytester_with_plugin
    pytester.makepyfile(
        """
import pytest

@pytest.mark.genai
@pytest.mark.parametrize("x", [1, 2, 3])
def test_param(x):
    assert x > 0
"""
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(["*MLflow GenAI Test Results*"])
    result.stdout.fnmatch_lines(["*3 passed*"])


def test_no_marker_no_summary(pytester_with_plugin):
    pytester, _ = pytester_with_plugin
    pytester.makepyfile(
        """
def test_plain():
    assert True
"""
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
    assert "MLflow GenAI Test Results" not in result.stdout.str()


def test_failed_test_outcome(pytester_with_plugin):
    pytester, _ = pytester_with_plugin
    pytester.makepyfile(
        """
import pytest

@pytest.mark.genai
def test_will_fail():
    assert False

@pytest.mark.genai
def test_will_pass():
    assert True
"""
    )
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*1 passed*1 failed*"])
