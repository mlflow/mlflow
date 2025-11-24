from unittest.mock import Mock, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.deepeval import get_judge


@pytest.fixture
def mock_deepeval_metric():
    metric = Mock()
    metric.score = 0.8
    metric.reason = "Test reason"
    metric.threshold = 0.5
    metric.is_successful.return_value = True
    metric.measure = Mock()
    return metric


@pytest.fixture
def mock_deepeval_metric_class(mock_deepeval_metric):
    with patch("mlflow.genai.scorers.deepeval.get_metric_class") as mock:
        mock.return_value = Mock(return_value=mock_deepeval_metric)
        yield mock


@pytest.fixture
def mock_deepeval_model():
    with patch("mlflow.genai.scorers.deepeval.create_deepeval_model") as mock:
        mock.return_value = Mock()
        yield mock


def test_deepeval_scorer_returns_correct_feedback_structure(
    mock_deepeval_metric_class, mock_deepeval_model
):
    judge = get_judge("AnswerRelevancy")
    result = judge(inputs="What is MLflow?", outputs="MLflow is a platform")

    assert isinstance(result, Feedback)
    assert result.name == "AnswerRelevancy"
    assert result.value == CategoricalRating.YES
    assert result.rationale == "Test reason"
    assert result.metadata["score"] == 0.8
    assert result.metadata["threshold"] == 0.5
    assert result.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert result.source.source_id == "deepeval/AnswerRelevancy"


def test_deepeval_scorer_handles_failure_correctly(mock_deepeval_metric_class, mock_deepeval_model):
    metric = mock_deepeval_metric_class.return_value.return_value
    metric.is_successful.return_value = False
    metric.score = 0.3

    judge = get_judge("AnswerRelevancy")
    result = judge(inputs="What is MLflow?", outputs="Unrelated answer")

    assert result.value == CategoricalRating.NO
    assert result.metadata["score"] == 0.3


def test_metric_kwargs_passed_to_deepeval_metric(mock_deepeval_metric_class, mock_deepeval_model):
    get_judge("AnswerRelevancy", threshold=0.9, include_reason=True, custom_param="value")

    metric_class_mock = mock_deepeval_metric_class.return_value
    call_kwargs = metric_class_mock.call_args[1]
    assert call_kwargs["threshold"] == 0.9
    assert call_kwargs["include_reason"] is True
    assert call_kwargs["custom_param"] == "value"
    assert call_kwargs["verbose_mode"] is False
    assert call_kwargs["async_mode"] is False


def test_deepeval_scorer_returns_error_feedback_on_exception(
    mock_deepeval_metric_class, mock_deepeval_model
):
    metric = mock_deepeval_metric_class.return_value.return_value
    metric.measure.side_effect = RuntimeError("Test error")

    judge = get_judge("AnswerRelevancy")
    result = judge(inputs="What is MLflow?", outputs="Test output")

    assert isinstance(result, Feedback)
    assert result.name == "AnswerRelevancy"
    assert result.value is None
    assert result.error is not None
    assert result.error.error_code == "RuntimeError"
    assert result.error.error_message == "Test error"
    assert result.source.source_type == AssessmentSourceType.LLM_JUDGE
