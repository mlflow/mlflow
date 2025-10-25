from unittest import mock

import pandas as pd
import pytest

import mlflow
from mlflow.entities import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.utils import _convert_to_eval_set
from mlflow.genai.scorers.base import Scorer, scorer, validate_feedback_names_unique
from mlflow.genai.scorers.builtin_scorers import (
    Correctness,
    ExpectationsGuidelines,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalSufficiency,
    get_all_scorers,
)
from mlflow.genai.scorers.validation import valid_data_for_builtin_scorers, validate_scorers

from tests.genai.conftest import databricks_only


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
            RelevanceToQuery(),
            Correctness(),
            Guidelines(guidelines=["Be polite", "Be kind"]),
            custom_scorer,
        ]
    )

    assert len(scorers) == 4
    assert all(isinstance(scorer, Scorer) for scorer in scorers)


def test_validate_scorers_empty_list():
    assert validate_scorers([]) == []


@databricks_only
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


def test_validate_scorers_invalid_all_scorers():
    with pytest.raises(MlflowException, match="The `scorers` argument must be a list") as e:
        validate_scorers([1, 2, 3])
    assert "an invalid item with type: int" in str(e.value)

    # Special case 1: List of list of all scorers
    with pytest.raises(MlflowException, match="The `scorers` argument must be a list") as e:
        validate_scorers([get_all_scorers()])

    assert "an invalid item with type: list" in str(e.value)
    assert "Hint: Use `scorers=get_all_scorers()` to pass all" in str(e.value)

    # Special case 2: List of list of all scorers + custom scorers
    with pytest.raises(MlflowException, match="The `scorers` argument must be a list") as e:
        validate_scorers([get_all_scorers(), RelevanceToQuery(), Correctness()])

    assert "an invalid item with type: list" in str(e.value)
    assert "Hint: Use `scorers=[*get_all_scorers(), scorer1, scorer2]` to pass all" in str(e.value)

    # Special case 3: List of classes (not instances)
    with pytest.raises(MlflowException, match="The `scorers` argument must be a list") as e:
        validate_scorers([RelevanceToQuery])

    assert "Correct way to pass scorers is `scorers=[RelevanceToQuery()]`." in str(e.value)


def test_validate_data(mock_logger, sample_rag_trace):
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
            "trace": [sample_rag_trace, sample_rag_trace],
        }
    )

    converted_date = _convert_to_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            RelevanceToQuery(),
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

    converted_date = _convert_to_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            RelevanceToQuery(),
            RetrievalSufficiency(),  # requires expected_response in expectations
            ExpectationsGuidelines(),  # requires guidelines in expectations
        ],
    )
    mock_logger.info.assert_not_called()


def test_global_guidelines_do_not_require_expectations(mock_logger):
    """Test that expectations are unwrapped and validated properly"""
    data = pd.DataFrame(
        {
            "inputs": [{"question": "input1"}, {"question": "input2"}],
            "outputs": ["output1", "output2"],
        }
    )
    converted_date = _convert_to_eval_set(data)
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

    converted_date = _convert_to_eval_set(data)
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

    converted_date = _convert_to_eval_set(data)

    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            RelevanceToQuery(),
            RetrievalGroundedness(),
            Guidelines(guidelines=["Be polite", "Be kind"]),
        ],
    )

    mock_logger.info.assert_called_once()
    msg = mock_logger.info.call_args[0][0]
    assert " - `outputs` column is required by [relevance_to_query, guidelines]." in msg
    assert " - `trace` column is required by [retrieval_groundedness]." in msg


def test_validate_data_with_trace(mock_logger):
    # When a trace is provided, the inputs, outputs, and retrieved_context are
    # inferred from the trace.
    with mlflow.start_span() as span:
        span.set_inputs({"question": "What is the capital of France?"})
        span.set_outputs("Paris")

    trace = mlflow.get_trace(span.trace_id)
    data = [{"trace": trace}, {"trace": trace}]

    converted_date = _convert_to_eval_set(data)
    valid_data_for_builtin_scorers(
        data=converted_date,
        builtin_scorers=[
            RelevanceToQuery(),
            RetrievalGroundedness(),
            Guidelines(guidelines=["Be polite", "Be kind"]),
        ],
    )
    mock_logger.info.assert_not_called()


def test_validate_data_with_predict_fn(mock_logger):
    data = pd.DataFrame({"inputs": [{"question": "input1"}, {"question": "input2"}]})

    converted_date = _convert_to_eval_set(data)

    valid_data_for_builtin_scorers(
        data=converted_date,
        predict_fn=lambda x: x,
        builtin_scorers=[
            # Requires "outputs" but predict_fn will provide it
            Guidelines(guidelines=["Be polite", "Be kind"]),
            # Requires "retrieved_context" but predict_fn will provide it
            RelevanceToQuery(),
        ],
    )

    mock_logger.info.assert_not_called()


def test_validate_feedback_names_unique_with_unique_names():
    feedbacks = [
        Feedback(name="metric_1", value=True),
        Feedback(name="metric_2", value=1.0),
        Feedback(name="metric_3", value="good"),
    ]

    # Should not raise an exception
    validate_feedback_names_unique(feedbacks, "test_scorer")


def test_validate_feedback_names_unique_with_duplicate_names():
    feedbacks = [
        Feedback(name="metric_1", value=True),
        Feedback(name="metric_2", value=1.0),
        Feedback(name="metric_1", value="duplicate"),  # Duplicate name
    ]

    with pytest.raises(
        MlflowException, match="Cannot register scorer 'test_scorer' because it returns multiple"
    ) as exc_info:
        validate_feedback_names_unique(feedbacks, "test_scorer")

    error_msg = str(exc_info.value)
    assert "duplicate names" in error_msg
    assert "metric_1" in error_msg
    assert "Each Feedback in the returned list must have a unique name" in error_msg


def test_validate_feedback_names_unique_with_all_default_names():
    # Both use default name "feedback"
    feedbacks = [
        Feedback(value=True),
        Feedback(value=1.0),
    ]

    with pytest.raises(
        MlflowException, match="Cannot register scorer 'test_scorer' because it returns multiple"
    ) as exc_info:
        validate_feedback_names_unique(feedbacks, "test_scorer")

    error_msg = str(exc_info.value)
    assert "duplicate names" in error_msg
    assert "feedback" in error_msg


def test_validate_feedback_names_unique_with_empty_list():
    # Should not raise an exception
    validate_feedback_names_unique([], "test_scorer")


def test_validate_feedback_names_unique_with_single_feedback():
    feedbacks = [Feedback(name="metric_1", value=True)]

    # Should not raise an exception
    validate_feedback_names_unique(feedbacks, "test_scorer")


@pytest.fixture
def mock_databricks_env():
    """Mock Databricks environment for scorer registration tests."""
    with (
        mock.patch(
            "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
        ),
        mock.patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer"),
    ):
        yield


def test_register_scorer_with_duplicate_feedback_names_fails(mock_databricks_env):
    @scorer
    def scorer_with_duplicates(outputs: str):
        return [
            Feedback(name="score", value=True),
            Feedback(name="score", value=1.0),  # Duplicate name
        ]

    with pytest.raises(
        MlflowException, match="Cannot register scorer 'scorer_with_duplicates'"
    ) as exc_info:
        scorer_with_duplicates._check_can_be_registered()

    error_msg = str(exc_info.value)
    assert "duplicate names" in error_msg


def test_register_scorer_with_default_feedback_names_fails(mock_databricks_env):
    @scorer
    def scorer_unnamed_feedbacks(outputs: str):
        return [
            Feedback(value=True, rationale="Good"),
            Feedback(value=1, rationale="ok"),
        ]

    with pytest.raises(
        MlflowException, match="Cannot register scorer 'scorer_unnamed_feedbacks'"
    ) as exc_info:
        scorer_unnamed_feedbacks._check_can_be_registered()

    error_msg = str(exc_info.value)
    assert "duplicate names" in error_msg
    assert "feedback" in error_msg


def test_register_scorer_with_unique_feedback_names_succeeds(mock_databricks_env):
    @scorer
    def scorer_with_unique_names(outputs: str):
        return [
            Feedback(name="relevance", value=True, rationale="Relevant"),
            Feedback(name="tone", value="professional", rationale="Professional tone"),
            Feedback(name="length", value=150, rationale="Good length"),
        ]

    # Should not raise an exception during validation
    scorer_with_unique_names._check_can_be_registered()


def test_register_scorer_with_single_feedback_succeeds(mock_databricks_env):
    @scorer
    def scorer_with_single_feedback(outputs: str):
        return Feedback(value=True, rationale="Good")

    # Should not raise an exception during validation
    scorer_with_single_feedback._check_can_be_registered()


def test_register_scorer_returning_primitive_succeeds(mock_databricks_env):
    @scorer
    def scorer_returning_bool(outputs: str) -> bool:
        return True

    # Should not raise an exception during validation
    scorer_returning_bool._check_can_be_registered()
