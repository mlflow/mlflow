from unittest import mock

import numpy as np
import pytest

from mlflow.entities.assessment import Feedback
from mlflow.genai.evaluation.entities import EvalItem, EvalResult, ScorerStat
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
        # NumPy scalars
        (np.int64(5), 5.0),
        (np.float32(0.5), 0.5),
        (np.bool_(True), 1.0),
    ],
)
def test_cast_numeric_values(value, expected_float):
    assessment = Feedback(name="test", value=value)
    assert _cast_assessment_value_to_float(assessment) == expected_float


@pytest.mark.parametrize("value", ["pass", ["good", "safe"], {"rating": "good"}])
def test_non_aggregatable_output_warns_agent(value):
    assessment = Feedback(name="quality", value=value)

    with mock.patch("mlflow.agent.hint.maybe_warn_agent") as warn:
        assert _cast_assessment_value_to_float(assessment) is None

    warn.assert_called_once_with(
        "non-aggregatable-scorer-output",
        f"Scorer 'quality' returned a value of type {type(value).__name__!r}, which is not "
        "included in aggregated metrics; return a boolean, number, or 'yes'/'no' instead.",
    )


def test_compute_aggregated_metrics_includes_extra_aggregations():
    scorers = [
        Scorer(name="default"),
        Scorer(name="explicit", aggregations=["max"]),
        Scorer(name="already_computed", aggregations=["mean"]),
    ]
    extra_aggregations = {"default": ["min"], "explicit": ["p90"], "already_computed": ["mean"]}
    eval_results = [
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[
                Feedback(name="default", value=value),
                Feedback(name="explicit", value=value),
                Feedback(name="already_computed", value=value),
            ],
        )
        for value in [0.2, 0.8]
    ]

    result = compute_aggregated_metrics(eval_results, scorers, extra_aggregations)

    assert result == {
        "default/mean": pytest.approx(0.5),
        "default/min": pytest.approx(0.2),
        "explicit/max": pytest.approx(0.8),
        "explicit/p90": pytest.approx(0.74),
        "already_computed/mean": pytest.approx(0.5),
    }
    # The scorer's own aggregations are not mutated.
    assert scorers[1].aggregations == ["max"]


def test_compute_aggregated_metrics_matches_scorer_name_containing_slash():
    scorers = [Scorer(name="team/latency"), Scorer(name="correctness", aggregations=["min"])]
    extra_aggregations = {"team/latency": ["max"]}
    eval_results = [
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[
                Feedback(name="team/latency", value=value),
                # Namespaced assessment names still resolve to the scorer by its own name.
                Feedback(name="judges/correctness", value=value),
            ],
        )
        for value in [1.0, 3.0]
    ]

    result = compute_aggregated_metrics(eval_results, scorers, extra_aggregations)

    assert result == {
        "team/latency/mean": pytest.approx(2.0),
        "team/latency/max": pytest.approx(3.0),
        "judges/correctness/min": pytest.approx(1.0),
    }


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # Some rows fail: the mean covers the scored rows, the count covers the failed ones.
        ([1.0, 0.0, None], {"scorer/mean": pytest.approx(0.5), "scorer/error_count": 1}),
        # No row fails: no count is logged.
        ([1.0, 0.0], {"scorer/mean": pytest.approx(0.5)}),
        # Every row fails: only the count is logged.
        ([None, None, None], {"scorer/error_count": 3}),
    ],
)
def test_compute_aggregated_metrics_logs_scorer_error_count(values, expected):
    scorer = Scorer(name="scorer")
    eval_results = [
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[
                Feedback(name="scorer", value=value)
                if value is not None
                else Feedback(name="scorer", error=Exception("judge timeout"))
            ],
        )
        for value in values
    ]
    scorer_stats = {"scorer": ScorerStat(failure_count=values.count(None), total_count=len(values))}

    result = compute_aggregated_metrics(eval_results, [scorer], scorer_stats=scorer_stats)

    assert result == expected


def test_compute_aggregated_metrics_counts_errors_per_scorer_not_per_feedback_name():
    # A scorer that returns several feedbacks records a failure under its own name.
    scorer = Scorer(name="multi")
    eval_results = [
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[Feedback(name="clarity", value=1.0), Feedback(name="safety", value=0.0)],
        ),
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[Feedback(name="multi", error=Exception("judge timeout"))],
        ),
    ]
    scorer_stats = {"multi": ScorerStat(failure_count=1, total_count=2)}

    result = compute_aggregated_metrics(eval_results, [scorer], scorer_stats=scorer_stats)

    assert result == {
        "clarity/mean": pytest.approx(1.0),
        "safety/mean": pytest.approx(0.0),
        "multi/error_count": 1,
    }


def test_compute_aggregated_metrics_custom_error_count_aggregation_wins():
    def error_count(x: list[float]) -> float:
        return 42.0

    scorer = Scorer(name="scorer", aggregations=[error_count])
    eval_results = [
        EvalResult(eval_item=_EVAL_ITEM, assessments=[Feedback(name="scorer", value=1.0)]),
        EvalResult(
            eval_item=_EVAL_ITEM,
            assessments=[Feedback(name="scorer", error=Exception("judge timeout"))],
        ),
    ]
    scorer_stats = {"scorer": ScorerStat(failure_count=1, total_count=2)}

    result = compute_aggregated_metrics(eval_results, [scorer], scorer_stats=scorer_stats)

    assert result == {"scorer/error_count": 42.0}
