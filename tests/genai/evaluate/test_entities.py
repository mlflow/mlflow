import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.genai.evaluation.entities import EvalItem, EvaluationResult
from mlflow.genai.evaluation.quality_criteria import QualityCriteriaRule
from mlflow.genai.judges import CategoricalRating


def test_eval_item_from_dataset_row_extracts_source():
    source = DatasetRecordSource(
        source_type=DatasetRecordSourceType.TRACE,
        source_data={"trace_id": "tr-123", "session_id": "session_1"},
    )

    row = {
        "inputs": {"question": "test"},
        "outputs": "answer",
        "expectations": {},
        "source": source,
    }

    eval_item = EvalItem.from_dataset_row(row)

    assert eval_item.source == source
    assert eval_item.source.source_data["session_id"] == "session_1"
    assert eval_item.inputs == {"question": "test"}
    assert eval_item.outputs == "answer"


def test_eval_item_from_dataset_row_handles_missing_source():
    row = {
        "inputs": {"question": "test"},
        "outputs": "answer",
        "expectations": {},
    }

    eval_item = EvalItem.from_dataset_row(row)

    assert eval_item.source is None
    assert eval_item.inputs == {"question": "test"}
    assert eval_item.outputs == "answer"


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


def test_no_rating_fails_cleanly_without_pass_if_hint():
    # "no" is a recognized rating, so it fails without nagging about pass_if.
    df = pd.DataFrame([{"scorer_a/value": "no", "scorer_a/rationale": None}])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "value='no'" in result.reason
    assert "pass_if" not in result.reason


def test_unrecognized_string_gets_pass_if_hint():
    # A non-yes/no string is not a recognized rating; surface the pass_if hint.
    df = pd.DataFrame([{"scorer_a/value": "pass", "scorer_a/rationale": None}])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "'pass'" in result.reason
    assert "pass_if" in result.reason


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


def test_numeric_value_without_pass_if_fails_loudly():
    df = pd.DataFrame([{"scorer_a/value": 0.7, "scorer_a/rationale": None}])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "pass_if" in result.reason


def test_numeric_value_rationale_does_not_suppress_pass_if_hint():
    df = pd.DataFrame([{"scorer_a/value": 0.7, "scorer_a/rationale": "looks good"}])
    result = EvaluationResult(run_id="r1", metrics={}, result_df=df)
    assert not result.passed
    assert "looks good" in result.reason
    assert "pass_if" in result.reason


def test_pass_if_predicate_gates_numeric_value():
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


def test_pass_if_raising_is_reported_not_propagated():
    df = pd.DataFrame([{"scorer_a/value": "weird", "scorer_a/rationale": None}])

    def boom(v):
        raise RuntimeError("bad predicate")

    result = EvaluationResult(
        run_id="r1", metrics={}, result_df=df, pass_criteria={"scorer_a": boom}
    )
    assert not result.passed
    assert "pass_if raised" in result.reason


def test_numpy_scalar_values():
    # Regression for np.bool_ / np.float64 scalars from DataFrame.iterrows().
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


@pytest.mark.parametrize(
    ("metrics", "expected_reason"),
    [
        ({"latency/p90": 1.5}, ""),
        ({"latency/p90": 2.0}, ""),
        ({"latency/p90": 4.35}, "quality criteria latency/p90 <= 2: got 4.35"),
        ({"latency/p90": 2.0000004}, "quality criteria latency/p90 <= 2: got 2.0000004"),
        ({}, "quality criteria latency/p90 <= 2: this evaluation produced no such metric"),
    ],
)
def test_quality_criteria_gate_passed(metrics, expected_reason):
    df = pd.DataFrame([{"latency/value": 1.0}, {"latency/value": 4.5}])
    result = EvaluationResult(
        run_id="r1",
        metrics=metrics,
        result_df=df,
        pass_criteria={"latency": lambda v: v <= 5},
        quality_criteria=[QualityCriteriaRule("latency", "p90", "<=", 2.0)],
    )
    assert result.passed is (expected_reason == "")
    assert result.reason == expected_reason


def test_quality_criteria_failures_are_listed_with_row_failures():
    df = pd.DataFrame([{"correctness/value": False}])
    result = EvaluationResult(
        run_id="r1",
        metrics={"correctness/mean": 0.0},
        result_df=None,
        quality_criteria=[QualityCriteriaRule("correctness", "mean", ">=", 0.9)],
    )
    assert result.reason == "quality criteria correctness/mean >= 0.9: got 0"

    result.result_df = df
    assert not result.passed
    assert result.reason.startswith("2 assertions failed:")
    assert "quality criteria correctness/mean >= 0.9: got 0" in result.reason


def test_eval_item_from_trace_takes_inputs_and_outputs_from_the_root_span():
    with mlflow.start_span(name="root") as span:
        span.set_inputs({"question": "q"})
        span.set_outputs({"answer": "a"})
    trace = mlflow.get_trace(span.trace_id)

    eval_item = EvalItem.from_trace(trace)

    assert eval_item.request_id == trace.info.trace_id
    assert eval_item.inputs == {"question": "q"}
    assert eval_item.outputs == {"answer": "a"}
    assert eval_item.trace is trace


def test_eval_item_from_trace_without_spans_has_no_inputs_or_outputs():
    with mlflow.start_span(name="root") as span:
        span.set_inputs({"question": "q"})
    trace = mlflow.get_trace(span.trace_id)
    trace.data.spans.clear()

    eval_item = EvalItem.from_trace(trace)

    assert eval_item.inputs is None
    assert eval_item.outputs is None
