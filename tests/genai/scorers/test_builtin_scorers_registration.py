"""Tests for builtin scorer registration validation with custom judge models."""

from unittest.mock import patch

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import RetrievalRelevance, Safety
from mlflow.genai.scorers.base import ScorerSamplingConfig


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_safety_with_non_databricks_model_cannot_register(_):
    """Test that Safety scorer with non-databricks model cannot be registered."""
    # OpenAI model
    scorer = Safety(model="openai:/gpt-4")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.register()

    # Anthropic model
    scorer = Safety(model="anthropic:/claude-3-opus")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.register()


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_retrieval_relevance_with_non_databricks_model_cannot_register(_):
    """Test that RetrievalRelevance scorer with non-databricks model cannot be registered."""
    scorer = RetrievalRelevance(model="openai:/gpt-4")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.register()


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
@patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer")
def test_safety_with_databricks_model_can_register(mock_add, _):
    """Test that Safety scorer with databricks model can be registered."""
    scorer = Safety(model="databricks:/my-judge-model")
    registered = scorer.register()

    assert registered.name == "safety"
    mock_add.assert_called_once()


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
@patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer")
def test_builtin_scorer_without_custom_model_can_register(mock_add, _):
    """Test that builtin scorers without custom models can be registered."""
    # Safety with default model (None)
    scorer = Safety()
    registered = scorer.register()
    assert registered.name == "safety"
    mock_add.assert_called_once()

    mock_add.reset_mock()

    # RetrievalRelevance with default model (None)
    scorer = RetrievalRelevance()
    registered = scorer.register()
    assert registered.name == "retrieval_relevance"
    mock_add.assert_called_once()


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_scorer_start_with_non_databricks_model_fails(_):
    """Test that start() method also validates model provider."""
    scorer = Safety(model="openai:/gpt-4")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_scorer_update_with_non_databricks_model_fails(_):
    """Test that update() method also validates model provider."""
    scorer = Safety(model="anthropic:/claude-3")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.3))


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_scorer_stop_with_non_databricks_model_fails(_):
    """Test that stop() method also validates model provider."""
    scorer = RetrievalRelevance(model="openai:/gpt-4")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.stop()


def test_non_databricks_backend_allows_any_model():
    """Test that non-databricks backends allow any model provider."""
    experiment_id = mlflow.create_experiment("test_any_model_allowed")

    # OpenAI model should work with MLflow backend
    scorer = Safety(model="openai:/gpt-4")
    registered = scorer.register(experiment_id=experiment_id)
    assert registered.name == "safety"

    # Anthropic model should work with MLflow backend
    scorer = RetrievalRelevance(model="anthropic:/claude-3-opus")
    registered = scorer.register(experiment_id=experiment_id)
    assert registered.name == "retrieval_relevance"

    # Clean up
    mlflow.delete_experiment(experiment_id)


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
def test_error_message_shows_actual_model(_):
    """Test that error message includes the actual model that was rejected."""
    model = "openai:/gpt-4-turbo"
    scorer = Safety(model=model)

    with pytest.raises(MlflowException, match=f"Got {model}") as exc_info:
        scorer.register()

    assert f"Got {model}" in str(exc_info.value)
