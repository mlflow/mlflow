"""Tests for the @mlflow.test pytest plugin."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import mlflow
from mlflow._assertions import session as _session
from mlflow._assertions.decorator import MLFLOW_TEST_ATTR
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_TEST

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
# Plugin sets current test identity only for @mlflow.test-marked tests
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
# End-to-end: run pytest in a subprocess and verify a single test run is
# actually created in the tracking store with the right tags.
# ---------------------------------------------------------------------------

_GENERATED_TEST = """
import mlflow
from mlflow.genai.scorers import scorer


@scorer
def always_pass(*, outputs):
    return True


@mlflow.test
def test_marked_one():
    mlflow.genai.evaluate(
        data=[{"inputs": {"text": "hi"}, "outputs": "hi"}],
        scorers=[always_pass],
    )


@mlflow.test
def test_marked_two():
    mlflow.genai.evaluate(
        data=[{"inputs": {"text": "bye"}, "outputs": "bye"}],
        scorers=[always_pass],
    )
"""


def _run_pytest(tmp_path: Path, file_name: str) -> tuple[subprocess.CompletedProcess, str]:
    """Run pytest on ``file_name`` in a subprocess against a sqlite store.

    Returns the completed process and the tracking URI so the caller can
    inspect the runs that were actually persisted.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    env = {**os.environ, "MLFLOW_TRACKING_URI": tracking_uri}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", file_name, "-p", "no:cacheprovider", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, tracking_uri


def _test_runs(tracking_uri: str):
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(experiment_ids=experiment_ids) if experiment_ids else []
    return [r for r in runs if r.data.tags.get(MLFLOW_RUN_TYPE) == MLFLOW_RUN_TYPE_TEST]


def test_pytest_run_creates_single_mlflow_run(tmp_path: Path):
    test_file = tmp_path / "test_generated.py"
    test_file.write_text(textwrap.dedent(_GENERATED_TEST))

    result, tracking_uri = _run_pytest(tmp_path, test_file.name)
    assert result.returncode == 0, result.stdout + result.stderr

    # Two @mlflow.test cases share one session-scoped run, not one run each.
    test_runs = _test_runs(tracking_uri)
    assert len(test_runs) == 1
    assert test_runs[0].info.status == "FINISHED"
    assert _session.TAG_SESSION_ID in test_runs[0].data.tags


def test_unmarked_pytest_run_creates_no_test_run(tmp_path: Path):
    test_file = tmp_path / "test_plain.py"
    test_file.write_text("def test_plain():\n    assert True\n")

    result, tracking_uri = _run_pytest(tmp_path, test_file.name)
    assert result.returncode == 0, result.stdout + result.stderr

    # No @mlflow.test marker -> the plugin never opens a run.
    assert _test_runs(tracking_uri) == []
