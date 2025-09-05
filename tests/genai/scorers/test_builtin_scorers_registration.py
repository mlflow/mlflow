"""Tests for builtin scorer registration validation with custom judge models."""

from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import RetrievalRelevance, Safety, Scorer
from mlflow.genai.scorers.base import ScorerSamplingConfig


@pytest.fixture
def mock_databricks_tracking_uri() -> Iterator[mock.Mock]:
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
    ) as mock_uri:
        yield mock_uri


@pytest.mark.parametrize(
    ("scorer_class", "model"),
    [
        (Safety, "openai:/gpt-4"),
        (Safety, "anthropic:/claude-3-opus"),
        (RetrievalRelevance, "openai:/gpt-4"),
        (RetrievalRelevance, "anthropic:/claude-3"),
    ],
)
def test_non_databricks_model_cannot_register(
    scorer_class: type[Scorer], model: str, mock_databricks_tracking_uri: mock.Mock
):
    scorer = scorer_class(model=model)
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.register()
    mock_databricks_tracking_uri.assert_called()


def test_safety_with_databricks_model_can_register(mock_databricks_tracking_uri: mock.Mock):
    with mock.patch(
        "mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer"
    ) as mock_add:
        scorer = Safety(model="databricks:/my-judge-model")
        registered = scorer.register()

    assert registered.name == "safety"
    mock_add.assert_called_once()
    mock_databricks_tracking_uri.assert_called()


def test_builtin_scorer_without_custom_model_can_register(mock_databricks_tracking_uri: mock.Mock):
    with mock.patch(
        "mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer"
    ) as mock_add:
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
    mock_databricks_tracking_uri.assert_called()


def test_scorer_start_with_non_databricks_model_fails(mock_databricks_tracking_uri: mock.Mock):
    scorer = Safety(model="openai:/gpt-4")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
    mock_databricks_tracking_uri.assert_called()


def test_scorer_update_with_non_databricks_model_fails(mock_databricks_tracking_uri: mock.Mock):
    scorer = Safety(model="anthropic:/claude-3")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.3))
    mock_databricks_tracking_uri.assert_called()


def test_scorer_stop_with_non_databricks_model_fails(mock_databricks_tracking_uri: mock.Mock):
    scorer = RetrievalRelevance(model="openai:/gpt-4")
    with pytest.raises(
        MlflowException, match="The scorer's judge model must use Databricks as a model provider"
    ):
        scorer.stop()
    mock_databricks_tracking_uri.assert_called()


@pytest.mark.parametrize(
    ("scorer_class", "model", "expected_name"),
    [
        (Safety, "openai:/gpt-4", "safety"),
        (Safety, "anthropic:/claude-3-opus", "safety"),
        (RetrievalRelevance, "openai:/gpt-4", "retrieval_relevance"),
        (RetrievalRelevance, "anthropic:/claude-3", "retrieval_relevance"),
    ],
)
def test_non_databricks_backend_allows_any_model(
    scorer_class: type[Scorer], model: str, expected_name: str, tmp_path: Path
):
    tracking_uri = f"sqlite:///{tmp_path}/test.db"
    mlflow.set_tracking_uri(tracking_uri)

    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri",
        return_value=tracking_uri,
    ) as mock_get_tracking_uri:
        experiment_id = mlflow.create_experiment("test_any_model_allowed")

        # Non-Databricks models should work with MLflow backend
        scorer = scorer_class(model=model)
        registered = scorer.register(experiment_id=experiment_id)
        assert registered.name == expected_name

        mock_get_tracking_uri.assert_called()


def test_error_message_shows_actual_model(mock_databricks_tracking_uri: mock.Mock):
    """Test that error message includes the actual model that was rejected."""
    model = "openai:/gpt-4-turbo"
    scorer = Safety(model=model)

    with pytest.raises(MlflowException, match=f"Got {model}"):
        scorer.register()

    mock_databricks_tracking_uri.assert_called()
