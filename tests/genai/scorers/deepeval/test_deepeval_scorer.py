from unittest.mock import Mock, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.deepeval import AnswerRelevancy, KnowledgeRetention, get_scorer


@pytest.fixture
def mock_deepeval_model():
    """Create a mock DeepEval model that satisfies DeepEval's validation."""
    from deepeval.models.base_model import DeepEvalBaseLLM

    class MockDeepEvalModel(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__(model_name="mock-model")

        def load_model(self):
            return self

        def generate(self, prompt: str, schema=None) -> str:
            return "mock response"

        async def a_generate(self, prompt: str, schema=None) -> str:
            return "mock response"

        def get_model_name(self) -> str:
            return "mock-model"

    return MockDeepEvalModel()


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


def test_multi_turn_metric_is_session_level_scorer(mock_deepeval_model):
    with patch(
        "mlflow.genai.scorers.deepeval.create_deepeval_model", return_value=mock_deepeval_model
    ):
        knowledge_retention = KnowledgeRetention()
        assert knowledge_retention.is_session_level_scorer is True

        answer_relevancy = AnswerRelevancy()
        assert answer_relevancy.is_session_level_scorer is False


def test_multi_turn_metric_requires_session_parameter(mock_deepeval_model):
    with patch(
        "mlflow.genai.scorers.deepeval.create_deepeval_model", return_value=mock_deepeval_model
    ):
        scorer = KnowledgeRetention()

        result = scorer(inputs="test", outputs="test")
        assert result.error is not None
        assert "requires 'session' parameter" in result.error.error_message


def test_multi_turn_metric_with_session(mock_deepeval_model):
    mock_conversational_test_case = Mock()

    with (
        patch(
            "mlflow.genai.scorers.deepeval.create_deepeval_model", return_value=mock_deepeval_model
        ),
        patch(
            "mlflow.genai.scorers.deepeval.map_session_to_deepeval_conversational_test_case",
            return_value=mock_conversational_test_case,
        ) as mock_map_session,
    ):
        mock_traces = [Mock(), Mock(), Mock()]

        scorer = KnowledgeRetention()

        # Mock the metric's behavior after it's created
        scorer._metric.score = 0.85
        scorer._metric.reason = "Good knowledge retention"
        scorer._metric.threshold = 0.7
        scorer._metric.is_successful = Mock(return_value=True)
        scorer._metric.measure = Mock()

        result = scorer(session=mock_traces)

        # Verify session mapping was called
        mock_map_session.assert_called_once_with(session=mock_traces, expectations=None)

        # Verify metric.measure was called with conversational test case
        scorer._metric.measure.assert_called_once_with(
            mock_conversational_test_case, _show_indicator=False
        )

        # Verify result
        assert isinstance(result, Feedback)
        assert result.name == "KnowledgeRetention"
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.85


def test_single_turn_metric_ignores_session_parameter():
    mock_test_case = Mock()
    mock_metric_instance = Mock()
    mock_metric_instance.score = 0.9
    mock_metric_instance.reason = "Highly relevant"
    mock_metric_instance.threshold = 0.7
    mock_metric_instance.is_successful.return_value = True

    with (
        patch("mlflow.genai.scorers.deepeval.create_deepeval_model"),
        patch(
            "mlflow.genai.scorers.deepeval.get_metric_class",
            return_value=Mock(return_value=mock_metric_instance),
        ),
        patch(
            "mlflow.genai.scorers.deepeval.map_scorer_inputs_to_deepeval_test_case",
            return_value=mock_test_case,
        ) as mock_map_inputs,
        patch(
            "mlflow.genai.scorers.deepeval.map_session_to_deepeval_conversational_test_case"
        ) as mock_map_session,
    ):
        mock_traces = [Mock(), Mock()]

        scorer = AnswerRelevancy()

        # Single-turn metric should use inputs/outputs even when session is provided
        result = scorer(inputs="question", outputs="answer", session=mock_traces)

        # Verify single-turn mapping was called, NOT session mapping
        mock_map_inputs.assert_called_once()
        mock_map_session.assert_not_called()

        # Verify result
        assert isinstance(result, Feedback)
        assert result.value == CategoricalRating.YES
