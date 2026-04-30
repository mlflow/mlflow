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

The plugin provides the following fixtures:

- ``mlflow_experiment_name``: Override to set the experiment name (default: ``"pytest"``).
- ``mlflow_run``: The active child run for the current test case.

CLI options:

- ``--mlflow-experiment``: Set the MLflow experiment name (default: ``"pytest"``).
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

import pytest

import mlflow

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


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
# Session-scoped: experiment + parent run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mlflow_experiment_name(request: pytest.FixtureRequest) -> str:
    """Return the experiment name. Override this fixture to customise."""
    return request.config.getoption("mlflow_experiment")


@pytest.fixture(scope="session")
def _mlflow_session_run(mlflow_experiment_name: str):
    """Create a session-scoped parent run that groups all test runs."""
    mlflow.set_experiment(mlflow_experiment_name)
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=f"pytest_{timestamp}") as parent_run:
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
        yield child_run
