from unittest.mock import Mock, patch

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.deepeval import get_scorer


def test_deepeval_scorer_with_exact_match_metric():
    scorer = get_scorer("ExactMatch")
    result = scorer(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "ExactMatch"
    assert result.value == CategoricalRating.YES
    assert result.metadata["score"] == 1.0
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "deepeval"
    assert result.source.source_type == AssessmentSourceType.CODE
    assert result.source.source_id is None


def test_deepeval_scorer_handles_failure_with_exact_match():
    scorer = get_scorer("ExactMatch")
    result = scorer(
        inputs="What is MLflow?",
        outputs="MLflow is different",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert result.value == CategoricalRating.NO
    assert result.metadata["score"] == 0.0
    assert result.metadata[FRAMEWORK_METADATA_KEY] == "deepeval"


def test_metric_kwargs_passed_to_deepeval_metric():
    with (
        patch("mlflow.genai.scorers.deepeval.get_metric_class") as mock_get_metric_class,
        patch("mlflow.genai.scorers.deepeval.create_deepeval_model") as mock_create_model,
    ):
        mock_metric_class = Mock()
        mock_metric_instance = Mock()
        mock_metric_instance.score = 0.8
        mock_metric_instance.reason = "Test"
        mock_metric_instance.threshold = 0.9
        mock_metric_instance.is_successful.return_value = True
        mock_metric_class.return_value = mock_metric_instance
        mock_get_metric_class.return_value = mock_metric_class
        mock_create_model.return_value = Mock()

        get_scorer("AnswerRelevancy", threshold=0.9, include_reason=True, custom_param="value")

        call_kwargs = mock_metric_class.call_args[1]
        assert call_kwargs["threshold"] == 0.9
        assert call_kwargs["include_reason"] is True
        assert call_kwargs["custom_param"] == "value"
        assert call_kwargs["verbose_mode"] is False
        assert call_kwargs["async_mode"] is False


def test_deepeval_scorer_returns_error_feedback_on_exception():
    with (
        patch("mlflow.genai.scorers.deepeval.get_metric_class") as mock_get_metric_class,
        patch("mlflow.genai.scorers.deepeval.create_deepeval_model") as mock_create_model,
    ):
        mock_metric_class = Mock()
        mock_metric_instance = Mock()
        mock_metric_instance.measure.side_effect = RuntimeError("Test error")
        mock_metric_class.return_value = mock_metric_instance
        mock_get_metric_class.return_value = mock_metric_class
        mock_create_model.return_value = Mock()

        scorer = get_scorer("AnswerRelevancy", model="openai:/gpt-4o")
        result = scorer(inputs="What is MLflow?", outputs="Test output")

        assert isinstance(result, Feedback)
        assert result.name == "AnswerRelevancy"
        assert result.value is None
        assert result.error is not None
        assert result.error.error_code == "RuntimeError"
        assert result.error.error_message == "Test error"
        assert result.source.source_type == AssessmentSourceType.LLM_JUDGE
        assert result.source.source_id == "openai:/gpt-4o"
