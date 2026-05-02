"""
MLflow GenAI pytest plugin.

Provides seamless integration between pytest and MLflow's GenAI evaluation framework.
Automatically manages MLflow experiments and runs so that each pytest session creates
a single parent run and each test case runs as a nested child run. Results from
``mlflow.genai.evaluate`` calls inside tests are automatically grouped under the
session run for easy review and aggregation.

Usage
-----
Install mlflow and add ``-p mlflow.genai.pytest_plugin`` to your pytest invocation or
``pytest.ini``::

    [pytest]
    addopts = -p mlflow.genai.pytest_plugin

Or register it in ``conftest.py``::

    pytest_plugins = ["mlflow.genai.pytest_plugin"]

Mark tests with ``@pytest.mark.genai`` to opt into automatic run management::

    @pytest.mark.genai
    def test_my_llm():
        result = mlflow.genai.evaluate(...)
        assert result.metrics["answer_similarity/mean"] > 0.8

The plugin provides the following fixtures:

- ``mlflow_experiment_name``: Override to set the experiment name (default: ``"pytest"``).
- ``mlflow_run``: The active child run for the current test case.
- ``mlflow_evaluate``: Convenience wrapper around ``mlflow.genai.evaluate()``.

CLI options:

- ``--mlflow-experiment``: Set the MLflow experiment name (default: ``"pytest"``).
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import mlflow

if TYPE_CHECKING:
    from typing import Any

_logger = logging.getLogger(__name__)

MLFLOW_RUN_TYPE_TAG = "mlflow.runType"
MLFLOW_RUN_TYPE_PYTEST = "pytest"
MLFLOW_TEST_OUTCOME_TAG = "mlflow.test.outcome"
MLFLOW_TEST_DURATION_TAG = "mlflow.test.duration"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TestResult:
    name: str
    outcome: str
    duration_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: str | None = None


# Public alias
TestResult = _TestResult


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("mlflow", "MLflow GenAI evaluation options")
    group.addoption(
        "--mlflow-experiment",
        dest="mlflow_experiment",
        default="pytest",
        help="MLflow experiment name for this test session (default: 'pytest').",
    )


# ---------------------------------------------------------------------------
# Marker registration and activation
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "genai: mark test for MLflow GenAI evaluation tracking")
    config._mlflow_genai_active = False
    config._mlflow_genai_results: list[_TestResult] = []
    config._mlflow_genai_parent_run_id: str | None = None
    config._mlflow_genai_experiment_name: str | None = None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("genai") is not None:
            config._mlflow_genai_active = True
            return


# ---------------------------------------------------------------------------
# Session-scoped: experiment + parent run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mlflow_experiment_name(request: pytest.FixtureRequest) -> str:
    """Return the experiment name. Override this fixture to customise."""
    return request.config.getoption("mlflow_experiment")


@pytest.fixture(scope="session")
def _mlflow_session_run(mlflow_experiment_name: str, request: pytest.FixtureRequest):
    """Create a session-scoped parent run that groups all test runs."""
    mlflow.set_experiment(mlflow_experiment_name)
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=f"pytest_{timestamp}") as parent_run:
        mlflow.set_tag(MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_PYTEST)
        request.config._mlflow_genai_parent_run_id = parent_run.info.run_id
        request.config._mlflow_genai_experiment_name = mlflow_experiment_name
        yield parent_run


# ---------------------------------------------------------------------------
# Test-scoped: child run per test
# ---------------------------------------------------------------------------


@pytest.fixture
def mlflow_run(_mlflow_session_run, request: pytest.FixtureRequest):
    """Provide a child MLflow run scoped to a single test case.

    When this fixture is active, calls to ``mlflow.genai.evaluate`` inside the
    test will automatically attach to this run.
    """
    test_name = request.node.name
    with mlflow.start_run(
        run_name=test_name,
        nested=True,
    ) as child_run:
        mlflow.set_tag(MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_PYTEST)
        # Log parametrize params if present
        if hasattr(request.node, "callspec"):
            mlflow.log_params(
                {k: str(v) for k, v in request.node.callspec.params.items()}
            )
        request.node._mlflow_run_id = child_run.info.run_id
        yield child_run


# ---------------------------------------------------------------------------
# Convenience fixture: mlflow_evaluate
# ---------------------------------------------------------------------------


@pytest.fixture
def mlflow_evaluate(request: pytest.FixtureRequest):
    """Convenience wrapper around ``mlflow.genai.evaluate()``.

    Captures evaluation metrics on the test node for the terminal summary.
    Accepts the same arguments as ``mlflow.genai.evaluate()``.
    """

    def _evaluate(*args: Any, **kwargs: Any):
        result = mlflow.genai.evaluate(*args, **kwargs)
        if not hasattr(request.node, "_mlflow_eval_metrics"):
            request.node._mlflow_eval_metrics = {}
        if result and hasattr(result, "metrics") and result.metrics:
            request.node._mlflow_eval_metrics.update(result.metrics)
        return result

    return _evaluate


# ---------------------------------------------------------------------------
# Hook: auto child run for @pytest.mark.genai tests
# ---------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    has_marker = item.get_closest_marker("genai") is not None
    uses_fixture = "mlflow_run" in getattr(item, "fixturenames", [])

    if not has_marker or uses_fixture:
        yield
        return

    # Auto-create a session run if one doesn't exist yet
    if not item.config._mlflow_genai_parent_run_id:
        experiment_name = item.config.getoption("mlflow_experiment")
        mlflow.set_experiment(experiment_name)
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        parent_run = mlflow.start_run(run_name=f"pytest_{timestamp}")
        mlflow.set_tag(MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_PYTEST)
        item.config._mlflow_genai_parent_run_id = parent_run.info.run_id
        item.config._mlflow_genai_experiment_name = experiment_name
        item.config._mlflow_auto_started_parent = True
    else:
        # Ensure we're in the parent run context
        parent_run = None

    test_name = item.name
    with mlflow.start_run(
        run_name=test_name,
        nested=True,
    ) as child_run:
        mlflow.set_tag(MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_PYTEST)
        if hasattr(item, "callspec"):
            mlflow.log_params(
                {k: str(v) for k, v in item.callspec.params.items()}
            )
        item._mlflow_run_id = child_run.info.run_id
        item._mlflow_start_time = time.monotonic()
        yield


# ---------------------------------------------------------------------------
# Hook: capture test outcome
# ---------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    has_marker = item.get_closest_marker("genai") is not None
    if not has_marker:
        return

    run_id = getattr(item, "_mlflow_run_id", None)
    duration = report.duration
    test_outcome = report.outcome  # "passed", "failed", "skipped"

    # Log outcome tags to the child run
    if run_id:
        try:
            client = mlflow.MlflowClient()
            client.set_tag(run_id, MLFLOW_TEST_OUTCOME_TAG, test_outcome)
            client.set_tag(run_id, MLFLOW_TEST_DURATION_TAG, f"{duration:.3f}")
        except Exception:
            _logger.debug("Failed to log test outcome tags", exc_info=True)

    # Accumulate results for terminal summary
    metrics = getattr(item, "_mlflow_eval_metrics", {})
    result = _TestResult(
        name=item.name,
        outcome=test_outcome,
        duration_s=duration,
        metrics=dict(metrics),
        run_id=run_id,
    )
    item.config._mlflow_genai_results.append(result)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config) -> None:
    if not getattr(config, "_mlflow_genai_active", False):
        return

    results: list[_TestResult] = getattr(config, "_mlflow_genai_results", [])
    if not results:
        return

    terminalreporter.section("MLflow GenAI Test Results")

    experiment_name = getattr(config, "_mlflow_genai_experiment_name", None)
    parent_run_id = getattr(config, "_mlflow_genai_parent_run_id", None)

    if experiment_name:
        terminalreporter.write_line(f"Experiment: {experiment_name}")
    if parent_run_id:
        terminalreporter.write_line(f"Parent Run: {parent_run_id}")
    terminalreporter.write_line("")

    # Summary counts
    passed = sum(1 for r in results if r.outcome == "passed")
    failed = sum(1 for r in results if r.outcome == "failed")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    terminalreporter.write_line(f"{passed} passed, {failed} failed, {skipped} skipped")
    terminalreporter.write_line("")

    # Table header
    header = f"{'Test':<50} {'Outcome':<10} {'Duration':<10} {'Metrics'}"
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))

    for r in results:
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in r.metrics.items()) if r.metrics else ""
        line = f"{r.name:<50} {r.outcome:<10} {r.duration_s:<10.3f} {metrics_str}"
        terminalreporter.write_line(line)

    # End auto-started parent run if needed
    if getattr(config, "_mlflow_auto_started_parent", False):
        try:
            mlflow.end_run()
        except Exception:
            _logger.debug("Failed to end auto-started parent run", exc_info=True)
