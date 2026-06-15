from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.judges import CategoricalRating
from mlflow.genai.scorers import scorer
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
    df = pd.DataFrame([
        {
            "scorer_a/value": True,
            "scorer_a/rationale": None,
            "scorer_b/value": False,
            "scorer_b/rationale": "bad output",
        }
    ])
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


def test_categorical_rating_value():
    df = pd.DataFrame([
        {"scorer_a/value": CategoricalRating.YES, "scorer_a/rationale": None},
        {"scorer_a/value": CategoricalRating.NO, "scorer_a/rationale": "nope"},
    ])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "scorer_a" in result.reason


def test_error_message_fails_with_detail():
    df = pd.DataFrame([
        {
            "scorer_a/value": None,
            "scorer_a/rationale": None,
            "scorer_a/error_message": "scorer blew up",
        }
    ])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "scorer blew up" in result.reason


def test_numeric_value_without_pass_when_fails_loudly():
    df = pd.DataFrame([{"scorer_a/value": 0.7, "scorer_a/rationale": None}])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "pass_when" in result.reason


def test_numeric_value_rationale_does_not_suppress_pass_when_hint():
    # A scorer rationale on a numeric value must augment, not replace, the
    # pass_when guidance, since this is exactly when the author needs it.
    df = pd.DataFrame([{"scorer_a/value": 0.7, "scorer_a/rationale": "looks good"}])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "looks good" in result.reason
    assert "pass_when" in result.reason


def test_pass_when_predicate_gates_numeric_value():
    df = pd.DataFrame([{"scorer_a/value": 0.7, "scorer_a/rationale": None}])

    lenient = EvaluationResult(
        run_id="r1", metrics={}, result_df=df, pass_criteria={"scorer_a": lambda v: v >= 0.6}
    )
    assert lenient.passed

    strict = EvaluationResult(
        run_id="r1", metrics={}, result_df=df, pass_criteria={"scorer_a": lambda v: v >= 0.8}
    )
    assert not strict.passed
    assert "scorer_a" in strict.reason


def test_pass_when_raising_is_reported_not_propagated():
    df = pd.DataFrame([{"scorer_a/value": "weird", "scorer_a/rationale": None}])

    def boom(v):
        raise RuntimeError("bad predicate")

    result = EvaluationResult(
        run_id="r1", metrics={}, result_df=df, pass_criteria={"scorer_a": boom}
    )
    assert not result.passed
    assert "pass_when raised" in result.reason


def test_numpy_scalar_values():
    # Regression for numpy scalars from DataFrame.iterrows() (np.bool_ / np.float64),
    # which the old type-matching rule mishandled.
    df = pd.DataFrame([
        {
            "flag/value": np.bool_(True),
            "score/value": np.float64(0.95),
        }
    ])
    result = EvaluationResult(
        run_id="r1", metrics={}, result_df=df, pass_criteria={"score": lambda v: v >= 0.9}
    )
    assert result.passed, result.reason

    df_fail = pd.DataFrame([{"flag/value": np.bool_(False)}])
    assert not EvaluationResult(run_id="r1", metrics={}, result_df=df_fail).passed


def test_sparse_columns_are_skipped():
    # Different rows run different scorers, so each row has NaN for the other's column.
    df = pd.DataFrame([
        {"scorer_a/value": True},
        {"scorer_b/value": True},
    ])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert result.passed, result.reason


def test_scorer_pass_when_is_exposed():
    @scorer(pass_when=lambda v: v >= 0.5)
    def my_score(outputs):
        return 0.6

    assert my_score.pass_when is not None
    assert my_score.pass_when(0.6) is True

    @scorer
    def plain(outputs):
        return True

    assert plain.pass_when is None


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
        [
            sys.executable,
            "-m",
            "pytest",
            test_file.name,
            "-p",
            "no:cacheprovider",
            # The plugin is opt-in (no pytest11 entry point).
            "-p",
            "mlflow.pytest.plugin",
            "-q",
        ],
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
