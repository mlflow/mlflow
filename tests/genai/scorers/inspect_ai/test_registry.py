import pytest
from inspect_ai import scorer as inspect_scorer

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.inspect_ai.registry import (
    get_scorer_factory,
    is_model_graded,
    resolve_factory_name,
)


@pytest.mark.parametrize(
    ("metric_name", "factory_name"),
    [
        ("ExactMatch", "exact"),
        ("Includes", "includes"),
        ("Match", "match"),
        ("Pattern", "pattern"),
        ("F1", "f1"),
        ("Answer", "answer"),
        ("ModelGradedFact", "model_graded_fact"),
        ("ModelGradedQA", "model_graded_qa"),
        ("exact", "exact"),
        ("choice", "choice"),
    ],
)
def test_resolve_factory_name(metric_name, factory_name):
    assert resolve_factory_name(metric_name) == factory_name
    assert get_scorer_factory(metric_name) is getattr(inspect_scorer, factory_name)


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("exact", False),
        ("ExactMatch", False),
        ("f1", False),
        ("model_graded_fact", True),
        ("ModelGradedQA", True),
        ("choice", False),
    ],
)
def test_is_model_graded(metric_name, expected):
    assert is_model_graded(metric_name) is expected


@pytest.mark.parametrize("metric_name", ["no_such_scorer", "CORRECT"])
def test_unknown_or_non_callable_names_raise(metric_name):
    with pytest.raises(MlflowException, match=f"Unknown Inspect AI scorer: '{metric_name}'"):
        get_scorer_factory(metric_name)
