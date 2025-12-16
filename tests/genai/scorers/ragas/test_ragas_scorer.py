from unittest.mock import patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.ragas import get_scorer


def test_ragas_scorer_with_exact_match_metric():
    judge = get_scorer("ExactMatch")
    result = judge(
        inputs="What is MLflow?",
        outputs="MLflow is a platform",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert isinstance(result, Feedback)
    assert result.name == "ExactMatch"
    assert result.value == 1.0
    assert result.source.source_type == AssessmentSourceType.CODE
    assert result.source.source_id == "ExactMatch"
    assert result.metadata["library"] == "ragas"


def test_ragas_scorer_handles_failure_with_exact_match():
    judge = get_scorer("ExactMatch")
    result = judge(
        inputs="What is MLflow?",
        outputs="MLflow is different",
        expectations={"expected_output": "MLflow is a platform"},
    )

    assert result.value == 0.0


def test_deterministic_metric_does_not_require_model():
    judge = get_scorer("ExactMatch")
    result = judge(
        outputs="test",
        expectations={"expected_output": "test"},
    )

    assert result.value == 1.0


def test_ragas_scorer_with_threshold_returns_categorical():
    judge = get_scorer("ExactMatch")
    judge._metric.threshold = 0.5
    with patch.object(judge._metric, "single_turn_score", return_value=0.8):
        result = judge(
            inputs="What is MLflow?",
            outputs="MLflow is a platform",
            expectations={"expected_output": "MLflow is a platform"},
        )

        assert result.value == CategoricalRating.YES
        assert result.metadata["score"] == 0.8
        assert result.metadata["threshold"] == 0.5


def test_ragas_scorer_with_threshold_returns_no_when_below():
    judge = get_scorer("ExactMatch")
    judge._metric.threshold = 0.5
    with patch.object(judge._metric, "single_turn_score", return_value=0.0):
        result = judge(
            inputs="What is MLflow?",
            outputs="Databricks is a company",
            expectations={"expected_output": "MLflow is a platform"},
        )

        assert result.value == CategoricalRating.NO
        assert result.metadata["score"] == 0.0
        assert result.metadata["threshold"] == 0.5


def test_ragas_scorer_without_threshold_returns_float():
    judge = get_scorer("ExactMatch")
    result = judge(
        outputs="test",
        expectations={"expected_output": "test"},
    )
    assert isinstance(result.value, float)
    assert result.value == 1.0
    assert "threshold" not in result.metadata


def test_ragas_scorer_returns_error_feedback_on_exception():
    judge = get_scorer("ExactMatch")

    with patch.object(judge._metric, "single_turn_score", side_effect=RuntimeError("Test error")):
        result = judge(inputs="What is MLflow?", outputs="Test output")

    assert isinstance(result, Feedback)
    assert result.name == "ExactMatch"
    assert result.value is None
    assert result.error is not None
    assert result.error.error_code == "RuntimeError"
    assert result.error.error_message == "Test error"
    assert result.source.source_type == AssessmentSourceType.CODE


def test_unknown_metric_raises_error():
    with pytest.raises(MlflowException, match="Unknown metric: 'NonExistentMetric'"):
        get_scorer("NonExistentMetric")


def test_missing_reference_parameter_returns_mlflow_error():
    judge = get_scorer("ContextPrecision")
    result = judge(inputs="What is MLflow?")
    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "ContextPrecision" in result.error.error_message  # metric name
    assert "trace with retrieval spans" in result.error.error_message  # mlflow error message
