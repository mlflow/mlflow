"""Tests for ``EvaluationResult.passed`` and ``EvaluationResult.reason``."""

from __future__ import annotations

import pandas as pd

import mlflow
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers import scorer


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


@scorer
def always_pass(*, outputs) -> bool:
    return True


@mlflow.test
def test_e2e_evaluate_all_pass():
    result = mlflow.genai.evaluate(
        data=[{"inputs": {"text": "hello"}, "outputs": "hello"}],
        scorers=[always_pass],
    )
    assert result.passed, result.reason
