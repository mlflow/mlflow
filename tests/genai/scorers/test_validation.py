from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.genai.evaluation.utils import _convert_to_legacy_eval_set
from mlflow.genai.scorers.base import Scorer, scorer
from mlflow.genai.scorers.builtin_scorers import (
    Correctness,
    ExpectationsGuidelines,
    Guidelines,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
)
from mlflow.genai.scorers.validation import valid_data_for_builtin_scorers, validate_scorers


@pytest.fixture
def mock_logger():
    with mock.patch("mlflow.genai.scorers.validation._logger") as mock_logger:
        yield mock_logger


def test_validate_scorers_valid():
    @scorer
    def custom_scorer(inputs, outputs):
        return 1.0

    scorers = validate_scorers(
        [
            RetrievalRelevance(),
            Correctness(),
            Guidelines(guidelines=["Be polite", "Be kind"]),
            custom_scorer,
        ]
    )

    assert len(scorers) == 4
    assert all(isinstance(scorer, Scorer) for scorer in scorers)


def test_validate_scorers_legacy_metric():
    from databricks.agents.evals import metric

    @metric
    def legacy_metric_1(request, response):
        return 1.0

    @metric
    def legacy_metric_2(request, response):
        return 1.0

    with mock.patch("mlflow.genai.scorers.validation._logger") as mock_logger:
        scorers = validate_scorers([legacy_metric_1, legacy_metric_2])

    assert len(scorers) == 2
    mock_logger.warning.assert_called_once()
    assert "legacy_metric_1" in mock_logger.warning.call_args[0][0]


def test_validate_data(mock_logger, sample_rag_trace):
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
            "trace": [sample_rag_trace, sample_rag_trace],
        }
    )

    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            RetrievalRelevance(),
            RetrievalGroundedness(),
            Guidelines(guidelines=["Be polite", "Be kind"]),
        ],
    )
    mock_logger.info.assert_not_called()


def test_validate_data_with_expectations(mock_logger, sample_rag_trace):
    """Test that expectations are unwrapped and validated properly"""
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
            "trace": [sample_rag_trace, sample_rag_trace],
            "expectations": [
                {"expected_response": "response1", "guidelines": ["Be polite", "Be kind"]},
                {"expected_response": "response2", "guidelines": ["Be nice", "Be strong"]},
            ],
        }
    )

    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            RetrievalRelevance(),
            RetrievalSufficiency(),  # requires expected_response in expectations
            ExpectationsGuidelines(),  # requires guidelines in expectations
        ],
    )
    mock_logger.info.assert_not_called()


def test_global_guideline_adherence_does_not_require_expectations(mock_logger):
    """Test that expectations are unwrapped and validated properly"""
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
        }
    )
    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[Guidelines(guidelines=["Be polite", "Be kind"])],
    )
    mock_logger.info.assert_not_called()


@pytest.mark.parametrize(
    "expectations",
    [
        {"expected_facts": [["fact1", "fact2"], ["fact3"]]},
        {"expected_response": ["expectation1", "expectation2"]},
    ],
)
def test_validate_data_with_correctness(expectations, mock_logger):
    """Correctness scorer requires one of expected_facts or expected_response"""
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
            "expectations": [expectations, expectations],
        }
    )

    converted_date = _convert_to_legacy_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[Correctness()],
    )

    valid_data_for_builtin_scorers(
        data=pd.DataFrame({"inputs": ["input1"], "outputs": ["output1"]}),
        builtin_scorers=[Correctness()],
    )

    mock_logger.info.assert_called_once()
    message = mock_logger.info.call_args[0][0]
    assert "expected_response or expected_facts" in message


def test_validate_data_missing_columns(mock_logger):
    data = pd.DataFrame({"inputs": [{"question": "input1"}, {"question": "input2"}]})

    converted_date = _convert_to_legacy_eval_set(data)

    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            RetrievalRelevance(),
            RetrievalGroundedness(),
            Guidelines(guidelines=["Be polite", "Be kind"]),
        ],
    )

    mock_logger.info.assert_called_once()
    msg = mock_logger.info.call_args[0][0]
    assert " - `outputs` column is required by [guidelines]." in msg
    assert " - `trace` column is required by [retrieval_relevance, retrieval_groundedness]." in msg


def test_validate_data_with_trace(mock_logger):
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
            RetrievalRelevance(),
            RetrievalGroundedness(),
            Guidelines(guidelines=["Be polite", "Be kind"]),
        ],
    )
    mock_logger.info.assert_not_called()


def test_validate_data_with_predict_fn(mock_logger):
    data = pd.DataFrame({"inputs": [{"question": "input1"}, {"question": "input2"}]})

    converted_date = _convert_to_legacy_eval_set(data)

    valid_data_for_builtin_scorers(
        data=converted_date,
        predict_fn=lambda x: x,
        builtin_scorers=[
            # Requires "outputs" but predict_fn will provide it
            Guidelines(guidelines=["Be polite", "Be kind"]),
            # Requires "retrieved_context" but predict_fn will provide it
            RetrievalRelevance(),
        ],
    )

    mock_logger.info.assert_not_called()
