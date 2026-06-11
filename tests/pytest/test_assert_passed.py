"""Tests for ``EvaluationResult.assert_passed()``."""

from __future__ import annotations

import pandas as pd
import pytest

import mlflow
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers import scorer


def test_all_passing():
    df = pd.DataFrame([
        {"scorer_a/value": True, "scorer_a/rationale": None},
        {"scorer_a/value": True, "scorer_a/rationale": None},
    ])
    EvaluationResult(run_id="r1", metrics={}, result_df=df).assert_passed()


def test_with_failures():
    df = pd.DataFrame([{
        "scorer_a/value": True,
        "scorer_a/rationale": None,
        "scorer_b/value": False,
        "scorer_b/rationale": "bad output",
    }])
    with pytest.raises(AssertionError, match="scorer_b"):
        EvaluationResult(run_id="r1", metrics={}, result_df=df).assert_passed()


def test_string_yes_no():
    df = pd.DataFrame([
        {"scorer_a/value": "yes", "scorer_a/rationale": None},
        {"scorer_a/value": "no", "scorer_a/rationale": "failed check"},
    ])
    with pytest.raises(AssertionError, match="scorer_a"):
        EvaluationResult(run_id="r1", metrics={}, result_df=df).assert_passed()


def test_none_result_df():
    EvaluationResult(run_id="r1", metrics={}, result_df=None).assert_passed()


@scorer
def always_pass(*, outputs) -> bool:
    return True


@mlflow.test
def test_e2e_evaluate_all_pass():
    result = mlflow.genai.evaluate(
        data=[{"inputs": {"text": "hello"}, "outputs": "hello"}],
        scorers=[always_pass],
    )
    result.assert_passed()
