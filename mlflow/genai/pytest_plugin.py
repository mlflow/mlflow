"""
MLflow GenAI pytest plugin.

Provides integration between pytest and MLflow's GenAI evaluation framework.
Tests marked with ``@pytest.mark.mlflow`` or using the ``mlflow_run`` fixture
automatically run inside nested child MLflow runs. When a marked test is first
encountered, the plugin creates a single session-scoped parent run and groups
all subsequent child runs beneath it. Results from ``mlflow.genai.evaluate``
calls inside these tests are grouped under the session run for review.

Usage
-----
This plugin is **opt-in** — it is NOT loaded automatically when mlflow is installed.
Users must explicitly register it in one of the following ways:

Register in ``conftest.py``::

    pytest_plugins = ["mlflow.genai.pytest_plugin"]

Or add to ``pytest.ini`` / ``pyproject.toml``::

    [pytest]
    addopts = -p mlflow.genai.pytest_plugin

Or pass on the command line::

    pytest -p mlflow.genai.pytest_plugin

Mark tests with ``@pytest.mark.mlflow`` to opt into automatic run management::

    @pytest.mark.mlflow
    def test_my_llm():
        result = mlflow.genai.evaluate(...)
        assert result.metrics["answer_similarity/mean"] > 0.8

The plugin provides the following fixtures:

- ``mlflow_experiment_name``: Override to set the experiment name
  (checks ``--mlflow-experiment`` CLI, then ``MLFLOW_EXPERIMENT_NAME`` /
  ``MLFLOW_EXPERIMENT_ID`` env vars, then falls back to MLflow's default
  experiment).
- ``mlflow_run``: The active child run for the current test case.
- ``mlflow_evaluate``: Convenience wrapper around ``mlflow.genai.evaluate()``.

CLI options:

- ``--mlflow-experiment``: Set the MLflow experiment name (overrides env vars).
"""

from __future__ import annotations

import datetime
import logging
import os
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
MLFLOW_TEST_NAME_TAG = "mlflow.test.name"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    __test__ = False

    name: str
    outcome: str
    duration_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_experiment(config) -> None:
    """Set the active MLflow experiment based on CLI / env / default.

    Resolution order:
    1. ``--mlflow-experiment`` CLI option (always treated as experiment *name*)
    2. ``MLFLOW_EXPERIMENT_NAME`` env var  → ``set_experiment(experiment_name=...)``
    3. ``MLFLOW_EXPERIMENT_ID`` env var    → ``set_experiment(experiment_id=...)``
    4. No-op — let MLflow use its default experiment
    """
    cli_value = config.getoption("mlflow_experiment", default=None)
    if cli_value is not None:
        mlflow.set_experiment(experiment_name=cli_value)
        config._mlflow_genai_experiment_name = cli_value
        return

    env_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if env_name:
        mlflow.set_experiment(experiment_name=env_name)
        config._mlflow_genai_experiment_name = env_name
        return

    env_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if env_id:
        mlflow.set_experiment(experiment_id=env_id)
        config._mlflow_genai_experiment_name = env_id
        return

    # Fall back to MLflow's default experiment (no set_experiment call needed)
    config._mlflow_genai_experiment_name = None


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("mlflow", "MLflow GenAI evaluation options")
    group.addoption(
        "--mlflow-experiment",
        dest="mlflow_experiment",
        default=None,
        help="MLflow experiment name for this test session.",
    )


# ---------------------------------------------------------------------------
# Marker registration and activation
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mlflow: mark test for MLflow GenAI evaluation tracking")
    config._mlflow_genai_active = False
    config._mlflow_genai_results: list[TestResult] = []
    config._mlflow_genai_parent_run_id: str | None = None
    config._mlflow_genai_experiment_name: str | None = None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("mlflow") is not None:
            config._mlflow_genai_active = True
            return


# ---------------------------------------------------------------------------
# Session-scoped: experiment + parent run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mlflow_experiment_name(request: pytest.FixtureRequest) -> str | None:
    """Return the experiment name. Override this fixture to customise.

    Resolution order:
    1. ``--mlflow-experiment`` CLI option (if explicitly provided)
    2. ``MLFLOW_EXPERIMENT_NAME`` environment variable
    3. ``MLFLOW_EXPERIMENT_ID`` environment variable (resolved by MLflow)
    4. ``None`` — MLflow uses its default experiment
    """
    cli_value = request.config.getoption("mlflow_experiment")
    if cli_value is not None:
        return cli_value
    env_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if env_name:
        return env_name
    env_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if env_id:
        return env_id
    return None


@pytest.fixture(scope="session")
def _mlflow_session_run(mlflow_experiment_name: str | None, request: pytest.FixtureRequest):
    """Create a session-scoped parent run that groups all test runs."""
    if mlflow_experiment_name is not None:
        # CLI option or MLFLOW_EXPERIMENT_NAME — treat as name
        env_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        if env_id and mlflow_experiment_name == env_id:
            mlflow.set_experiment(experiment_id=env_id)
        else:
            mlflow.set_experiment(experiment_name=mlflow_experiment_name)
    # else: let MLflow use its default experiment

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
        mlflow.set_tag(MLFLOW_TEST_NAME_TAG, test_name)
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
# Hook: auto child run for @pytest.mark.mlflow tests
# ---------------------------------------------------------------------------


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    has_marker = item.get_closest_marker("mlflow") is not None
    uses_fixture = "mlflow_run" in getattr(item, "fixturenames", [])

    if not has_marker or uses_fixture:
        yield
        return

    # Auto-create a session run if one doesn't exist yet
    if not item.config._mlflow_genai_parent_run_id:
        _resolve_experiment(item.config)
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        parent_run = mlflow.start_run(run_name=f"pytest_{timestamp}")
        mlflow.set_tag(MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_PYTEST)
        item.config._mlflow_genai_parent_run_id = parent_run.info.run_id
        item.config._mlflow_auto_started_parent = True
    else:
        parent_run = None

    test_name = item.name
    with mlflow.start_run(
        run_name=test_name,
        nested=True,
    ) as child_run:
        mlflow.set_tag(MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_PYTEST)
        mlflow.set_tag(MLFLOW_TEST_NAME_TAG, test_name)
        if hasattr(item, "callspec"):
            mlflow.log_params(
                {k: str(v) for k, v in item.callspec.params.items()}
            )
        item._mlflow_run_id = child_run.info.run_id
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

    has_marker = item.get_closest_marker("mlflow") is not None
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
    result = TestResult(
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

    results: list[TestResult] = getattr(config, "_mlflow_genai_results", [])
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

    # Log summary metrics to the parent run
    if parent_run_id:
        total_duration = sum(r.duration_s for r in results)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0.0
        try:
            client = mlflow.MlflowClient()
            summary_metrics = {
                "test.pass_count": passed,
                "test.fail_count": failed,
                "test.skip_count": skipped,
                "test.pass_rate": pass_rate,
                "test.total_duration": total_duration,
            }
            for key, value in summary_metrics.items():
                client.log_metric(parent_run_id, key, value)
        except Exception:
            _logger.debug("Failed to log summary metrics to parent run", exc_info=True)

    # End auto-started parent run if needed
    if getattr(config, "_mlflow_auto_started_parent", False):
        try:
            mlflow.end_run()
        except Exception:
            _logger.debug("Failed to end auto-started parent run", exc_info=True)
