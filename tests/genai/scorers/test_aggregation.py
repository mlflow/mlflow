import pytest

from mlflow.entities.assessment import Feedback
from mlflow.genai.evaluation.entities import EvalItem, EvalResult
from mlflow.genai.judges.builtin import CategoricalRating
from mlflow.genai.scorers.aggregation import (
    _cast_assessment_value_to_float,
    compute_aggregated_metrics,
)
from mlflow.genai.scorers.base import Scorer

_EVAL_ITEM = EvalItem(
    request_id="dummy_request_id",
    inputs={"dummy_input": "dummy_input"},
    outputs="dummy_output",
    expectations={"dummy_expectation": "dummy_expectation"},
    tags={"test_tag": "test_value"},
    trace=None,
)


def test_compute_aggregated_metrics():
    scorer1 = Scorer(name="scorer1")  # Should default to ["mean"]
    scorer2 = Scorer(
        name="scorer2", aggregations=["mean", "min", "max", "median", "variance", "p90"]
    )

    eval_results = [
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[Feedback(name="scorer1", value=0.8), Feedback(name="scorer2", value=0.7)],
        ),
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[Feedback(name="scorer1", value=0.9)],
        ),
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[
                Feedback(name="scorer1", value=0.7),
                Feedback(name="scorer2", value=0.5),
                Feedback(name="scorer2", value=0.6),  # Multiple assessments from a scorer
            ],
        ),
        EvalResult(
            eval_item=_EVAL_ITEM,
            # Should filter out assessment without a value
            assessments=[Feedback(name="scorer1", error=Exception("Error"))],
        ),
    ]

    result = compute_aggregated_metrics(eval_results, [scorer1, scorer2])

    assert result["scorer1/mean"] == pytest.approx(0.8)
    assert result["scorer2/mean"] == pytest.approx(0.6)
    assert result["scorer2/min"] == pytest.approx(0.5)
    assert result["scorer2/max"] == pytest.approx(0.7)
    assert result["scorer2/median"] == pytest.approx(0.6)
    assert result["scorer2/variance"] == pytest.approx(0.00666666666)
    assert result["scorer2/p90"] == pytest.approx(0.68)


def test_compute_aggregated_metrics_custom_function():
    def custom_sum(x: list[float]) -> float:
        return sum(x)

    def custom_count(x: list[float]) -> float:
        return len(x)

    scorer = Scorer(name="scorer", aggregations=["mean", custom_sum, custom_count])
    eval_results = [
        EvalResult(eval_item=_EVAL_ITEM, assessments=[Feedback(name="scorer", value=0.8)]),
        EvalResult(eval_item=_EVAL_ITEM, assessments=[Feedback(name="scorer", value=0.9)]),
        EvalResult(eval_item=_EVAL_ITEM, assessments=[Feedback(name="scorer", value=0.7)]),
    ]
    result = compute_aggregated_metrics(eval_results, [scorer])

    assert result["scorer/mean"] == pytest.approx(0.8)
    assert result["scorer/custom_sum"] == pytest.approx(2.4)
    assert result["scorer/custom_count"] == pytest.approx(3)


def test_compute_aggregated_metrics_empty():
    scorer = Scorer(name="scorer", aggregations=["mean"])
    eval_results = []
    result = compute_aggregated_metrics(eval_results, [scorer])
    assert result == {}


def test_compute_aggregated_metrics_with_namespace():
    scorer = Scorer(name="scorer1", aggregations=["mean", "max"])
    eval_results = [
        EvalResult(eval_item=_EVAL_ITEM, assessments=[Feedback(name="foo/scorer1", value=1.0)]),
        EvalResult(eval_item=_EVAL_ITEM, assessments=[Feedback(name="foo/scorer1", value=2.0)]),
    ]

    result = compute_aggregated_metrics(eval_results, [scorer])
    assert result["foo/scorer1/mean"] == pytest.approx(1.5)
    assert result["foo/scorer1/max"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("value", "expected_float"),
    [
        (5, 5.0),
        (3.14, 3.14),
        (True, 1.0),
        (False, 0.0),
        (CategoricalRating.YES, 1.0),
        (CategoricalRating.NO, 0.0),
        ("yes", 1.0),
        ("no", 0.0),
        # Case-insensitive
        ("Yes", 1.0),
        ("No", 0.0),
    ],
)
def test_cast_numeric_values(value, expected_float):
    assessment = Feedback(name="test", value=value)
    assert _cast_assessment_value_to_float(assessment) == expected_float
