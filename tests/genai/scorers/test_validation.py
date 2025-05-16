from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.utils import _convert_to_legacy_eval_set
from mlflow.genai.scorers.base import BuiltInScorer, Scorer, scorer
from mlflow.genai.scorers.builtin_scorers import (
    chunk_relevance,
    context_sufficiency,
    correctness,
    global_guideline_adherence,
    groundedness,
)
from mlflow.genai.scorers.validation import valid_data_for_builtin_scorers, validate_scorers


def test_validate_scorers_valid():
    @scorer
    def custom_scorer(inputs, outputs):
        return 1.0

    builtin, custom = validate_scorers(
        [
            chunk_relevance(),
            correctness(),
            global_guideline_adherence(["Be polite", "Be kind"]),
            custom_scorer,
        ]
    )

    assert len(builtin) == 3
    assert all(isinstance(scorer, BuiltInScorer) for scorer in builtin)
    assert len(custom) == 1
    assert isinstance(custom[0], Scorer)


def test_validate_scorers_legacy_metric():
    from databricks.agents.evals import metric

    @metric
    def legacy_metric_1(request, response):
        return 1.0

    @metric
    def legacy_metric_2(request, response):
        return 1.0

    with mock.patch("mlflow.genai.scorers.validation._logger") as mock_logger:
        builtin, custom = validate_scorers([legacy_metric_1, legacy_metric_2])

    assert len(builtin) == 0
    assert len(custom) == 2
    mock_logger.warning.assert_called_once()
    assert "legacy_metric_1" in mock_logger.warning.call_args[0][0]


def test_validate_scorers_builtin_scorer_passed_as_function():
    with pytest.raises(MlflowException, match=r"A built-in scorer chunk_relevance"):
        validate_scorers([chunk_relevance])


def test_validate_data():
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
            "retrieved_context": [{"context": "context1"}, {"context": "context2"}],
        }
    )

    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            chunk_relevance(),
            groundedness(),
            global_guideline_adherence(["Be polite", "Be kind"]),
        ],
    )


@pytest.mark.parametrize(
    "expectations",
    [
        {
            "expectations": [
                {"expected_response": "response1"},
                {"expected_response": "response2"},
            ],
        },
    ],
)
def test_validate_data_with_expectations(expectations):
    """Test that expectations are unwrapped and validated properly"""
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
            "retrieved_context": ["context1", "context2"],
            **expectations,
        }
    )

    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            chunk_relevance(),
            context_sufficiency(),  # requires expected_response
        ],
    )


@pytest.mark.parametrize(
    "expectations",
    [
        {"expected_facts": [["fact1", "fact2"], ["fact3"]]},
        {"expected_response": ["expectation1", "expectation2"]},
    ],
)
def test_validate_data_with_correctness(expectations):
    """Correctness scorer requires one of expected_facts or expected_response"""
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
            **expectations,
        }
    )

    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[correctness()],
    )

    with pytest.raises(MlflowException, match=r"The input data is missing following") as e:
        valid_data_for_builtin_scorers(
            data=pd.DataFrame({"inputs": ["input1"], "outputs": ["output1"]}),
            builtin_scorers=[correctness()],
        )

    assert "expected_response or expected_facts" in str(e.value)


def test_validate_data_missing_columns():
    data = pd.DataFrame({"inputs": [{"question": "input1"}, {"question": "input2"}]})

    converted_date = _convert_to_legacy_eval_set(data)

    with pytest.raises(MlflowException, match=r"The input data is missing") as e:
        valid_data_for_builtin_scorers(
            data=converted_date,
            builtin_scorers=[
                chunk_relevance(),
                groundedness(),
                global_guideline_adherence(["Be polite", "Be kind"]),
            ],
        )

    assert " - `outputs` is required by [groundedness, guideline_adherence]." in str(e.value)
    assert " - `retrieved_context` is required by [chunk_relevance, groundedness]." in str(e.value)


def test_validate_data_with_trace():
    # When a trace is provided, the inputs, outputs, and retrieved_context are
    # inferred from the trace.
    with mlflow.start_span() as span:
        span.set_inputs({"question": "What is the capital of France?"})
        span.set_outputs("Paris")

    trace = mlflow.get_trace(span.trace_id)
    data = [{"trace": trace}, {"trace": trace}]

    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            chunk_relevance(),
            groundedness(),
            global_guideline_adherence(["Be polite", "Be kind"]),
        ],
    )


def test_validate_data_with_predict_fn():
    data = pd.DataFrame({"inputs": [{"question": "input1"}, {"question": "input2"}]})

    converted_date = _convert_to_legacy_eval_set(data)

    valid_data_for_builtin_scorers(
        data=converted_date,
        predict_fn=lambda x: x,
        builtin_scorers=[
            # Requires "outputs" but predict_fn will provide it
            global_guideline_adherence(["Be polite", "Be kind"]),
            # Requires "retrieved_context" but predict_fn will provide it
            chunk_relevance(),
        ],
    )
