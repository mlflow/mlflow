from unittest.mock import Mock, patch

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.ragas import get_judge


def test_ragas_scorer_with_exact_match_metric():
    judge = get_judge("ExactMatch")
    result = judge(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "ExactMatch"
    assert result.value == CategoricalRating.YES
    assert result.metadata["score"] == 1.0
    assert result.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert result.source.source_id == "ragas/ExactMatch"


def test_ragas_scorer_handles_failure_with_exact_match():
    judge = get_judge("ExactMatch")
    result = judge(
        inputs="What is MLflow?",
        outputs="MLflow is different",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert result.value == CategoricalRating.NO
    assert result.metadata["score"] == 0.0


def test_metric_kwargs_passed_to_ragas_metric():
    with patch("mlflow.genai.scorers.ragas.is_deterministic_metric") as mock_is_det:
        mock_is_det.return_value = False

        with patch("mlflow.genai.scorers.ragas.get_metric_class") as mock_get_metric_class:
            mock_metric_class = Mock()
            mock_metric_instance = Mock()
            mock_result = Mock()
            mock_result.score = 0.8
            mock_result.explanation = "Test explanation"
            mock_metric_instance.single_turn_score.return_value = mock_result
            mock_metric_instance.threshold = 0.9
            mock_metric_class.return_value = mock_metric_instance
            mock_get_metric_class.return_value = mock_metric_class

            with patch("mlflow.genai.scorers.ragas.create_ragas_model") as mock_create_model:
                mock_create_model.return_value = Mock()

                get_judge("Faithfulness", threshold=0.9, custom_param="value")

                call_kwargs = mock_metric_class.call_args[1]
                assert call_kwargs["threshold"] == 0.9
                assert call_kwargs["custom_param"] == "value"


def test_ragas_scorer_returns_error_feedback_on_exception():
    with patch("mlflow.genai.scorers.ragas.is_deterministic_metric") as mock_is_det:
        mock_is_det.return_value = False

        with patch("mlflow.genai.scorers.ragas.get_metric_class") as mock_get_metric_class:
            mock_metric_class = Mock()
            mock_metric_instance = Mock()
            mock_metric_instance.single_turn_score.side_effect = RuntimeError("Test error")
            mock_metric_class.return_value = mock_metric_instance
            mock_get_metric_class.return_value = mock_metric_class

            with patch("mlflow.genai.scorers.ragas.create_ragas_model") as mock_create_model:
                mock_create_model.return_value = Mock()

                judge = get_judge("Faithfulness")
                result = judge(inputs="What is MLflow?", outputs="Test output")

                assert isinstance(result, Feedback)
                assert result.name == "Faithfulness"
                assert result.value is None
                assert result.error is not None
                assert result.error.error_code == "RuntimeError"
                assert result.error.error_message == "Test error"
                assert result.source.source_type == AssessmentSourceType.LLM_JUDGE


def test_deterministic_metric_does_not_create_llm():
    with patch("mlflow.genai.scorers.ragas.create_ragas_model") as mock_create_model:
        judge = get_judge("ExactMatch")
        result = judge(
            outputs="test",
            expectations={"expected_output": "test"},
        )

        mock_create_model.assert_not_called()
        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 1.0


def test_ragas_scorer_with_threshold():
    with patch("mlflow.genai.scorers.ragas.is_deterministic_metric") as mock_is_det:
        mock_is_det.return_value = False

        with patch("mlflow.genai.scorers.ragas.get_metric_class") as mock_get_metric_class:
            mock_metric_class = Mock()
            mock_metric_instance = Mock()
            mock_metric_instance.single_turn_score.return_value = 0.6
            mock_metric_instance.threshold = 0.7
            mock_metric_class.return_value = mock_metric_instance
            mock_get_metric_class.return_value = mock_metric_class

            with patch("mlflow.genai.scorers.ragas.create_ragas_model") as mock_create_model:
                mock_create_model.return_value = Mock()

                judge = get_judge("Faithfulness")
                result = judge(inputs="test", outputs="test")

                assert result.value == CategoricalRating.NO
                assert result.metadata["score"] == 0.6
                assert result.metadata["threshold"] == 0.7
