from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mlflow.pytest import session as _session
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_TEST

# ---------------------------------------------------------------------------
# End-to-end: run pytest in a subprocess (its own session) and verify a single
# test run is created in the tracking store with the right tags. Marked tests
# run only here, never in this shared suite, so they don't leak an active run.
# ---------------------------------------------------------------------------

_GENERATED_TEST = """
import mlflow
from mlflow.pytest import session
from mlflow.genai.scorers import scorer


@scorer
def always_pass(*, outputs):
    return True


@mlflow.test
def test_marked_one():
    # Identity is the full test id; set for the running marked test.
    name, case_id = session.current_test()
    assert name.endswith("::test_marked_one")
    assert case_id is None
    mlflow.genai.evaluate(
        data=[{"inputs": {"text": "hi"}, "outputs": "hi"}],
        scorers=[always_pass],
    )


@mlflow.test()
def test_marked_called():
    # The called form @mlflow.test() works too.
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
    test_file.write_text(_GENERATED_TEST)

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


# ---------------------------------------------------------------------------
# Run status reflects only @mlflow.test outcomes.
# ---------------------------------------------------------------------------

_MARKED_FAILS = """
import mlflow
from mlflow.genai.scorers import scorer


@scorer
def always_pass(*, outputs):
    return True


@mlflow.test
def test_marked_fails():
    mlflow.genai.evaluate(
        data=[{"inputs": {"text": "hi"}, "outputs": "hi"}],
        scorers=[always_pass],
    )
    assert False
"""

_ONLY_UNMARKED_FAILS = """
import mlflow
from mlflow.genai.scorers import scorer


@scorer
def always_pass(*, outputs):
    return True


@mlflow.test
def test_marked_passes():
    mlflow.genai.evaluate(
        data=[{"inputs": {"text": "hi"}, "outputs": "hi"}],
        scorers=[always_pass],
    )


def test_unmarked_fails():
    assert False
"""


def test_run_marked_failed_when_a_marked_test_fails(tmp_path: Path):
    test_file = tmp_path / "test_marked_fails.py"
    test_file.write_text(_MARKED_FAILS)

    result, tracking_uri = _run_pytest(tmp_path, test_file.name)
    assert result.returncode != 0

    test_runs = _test_runs(tracking_uri)
    assert len(test_runs) == 1
    assert test_runs[0].info.status == "FAILED"


def test_run_finished_when_only_unmarked_test_fails(tmp_path: Path):
    test_file = tmp_path / "test_only_unmarked_fails.py"
    test_file.write_text(_ONLY_UNMARKED_FAILS)

    result, tracking_uri = _run_pytest(tmp_path, test_file.name)
    assert result.returncode != 0  # the unmarked test failed

    # Run status reflects only @mlflow.test outcomes -> the marked test passed.
    test_runs = _test_runs(tracking_uri)
    assert len(test_runs) == 1
    assert test_runs[0].info.status == "FINISHED"


# ---------------------------------------------------------------------------
# Parametrized marked test: case_id comes from pytest's callspec id, even when
# the param value contains brackets.
# ---------------------------------------------------------------------------

_PARAMETRIZED = """
import mlflow
import pytest
from mlflow.pytest import session
from mlflow.genai.scorers import scorer


@scorer
def always_pass(*, outputs):
    return True


@pytest.mark.parametrize("value", ["a", "[", "]"])
@mlflow.test
def test_param(value):
    name, case_id = session.current_test()
    assert "::test_param[" in name
    assert case_id == value
    mlflow.genai.evaluate(
        data=[{"inputs": {"text": value}, "outputs": value}],
        scorers=[always_pass],
    )
"""


def test_parametrized_marked_test_captures_case_id(tmp_path: Path):
    test_file = tmp_path / "test_param.py"
    test_file.write_text(_PARAMETRIZED)

    result, tracking_uri = _run_pytest(tmp_path, test_file.name)
    assert result.returncode == 0, result.stdout + result.stderr

    # All three cases share the one session run.
    assert len(_test_runs(tracking_uri)) == 1
