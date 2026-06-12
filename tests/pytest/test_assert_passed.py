"""Tests for ``EvaluationResult.passed`` / ``EvaluationResult.reason`` and the
``@mlflow.test`` trace tagging that the regression-test UI groups by.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd

from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.pytest.session import TAG_TEST_NAME
from mlflow.tracking import MlflowClient


def test_all_passing():
    df = pd.DataFrame([
        {"scorer_a/value": True, "scorer_a/rationale": None},
        {"scorer_a/value": True, "scorer_a/rationale": None},
    ])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert result.passed
    assert result.reason == ""


def test_with_failures():
    df = pd.DataFrame([{
        "scorer_a/value": True,
        "scorer_a/rationale": None,
        "scorer_b/value": False,
        "scorer_b/rationale": "bad output",
    }])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "scorer_b" in result.reason


def test_string_yes_no():
    df = pd.DataFrame([
        {"scorer_a/value": "yes", "scorer_a/rationale": None},
        {"scorer_a/value": "no", "scorer_a/rationale": "failed check"},
    ])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "scorer_a" in result.reason


def test_none_result_df():
    result = EvaluationResult(run_id="r1", metrics={}, result_df=None)
    assert result.passed


# ---------------------------------------------------------------------------
# End-to-end: a @mlflow.test runs in its own subprocess session, asserts on the
# evaluate() result, and the produced trace is tagged with the test identity.
# ---------------------------------------------------------------------------

_GENERATED_TEST = """
import mlflow
from mlflow.genai.scorers import scorer


@scorer
def always_pass(*, outputs):
    return True


@mlflow.test
def test_marked():
    result = mlflow.genai.evaluate(
        data=[{"inputs": {"text": "hi"}, "outputs": "hi"}],
        scorers=[always_pass],
    )
    assert result.passed, result.reason
"""


def test_evaluate_traces_tagged_with_test_identity(tmp_path: Path):
    test_file = tmp_path / "test_generated.py"
    test_file.write_text(textwrap.dedent(_GENERATED_TEST))

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file.name, "-p", "no:cacheprovider", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": tracking_uri},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    traces = client.search_traces(experiment_ids=experiment_ids) if experiment_ids else []
    assert traces
    assert all(t.info.tags.get(TAG_TEST_NAME, "").endswith("::test_marked") for t in traces)
