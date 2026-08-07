import numpy as np
import pandas as pd

from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.genai.evaluation.entities import EvalItem, EvaluationResult
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
