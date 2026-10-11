"""Tests for mlflow.genai.scorers.inspect_ai.models.

Verifies that MlflowInspectAIModel and create_inspectai_model correctly wrap
ScorerLLMClient, forward model_kwargs, and expose the expected interface.
"""
from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers.inspect_ai.models import MlflowInspectAIModel, create_inspectai_model


@pytest.fixture
def mock_scorer_llm_client():
    with patch("mlflow.genai.scorers.inspect_ai.models.ScorerLLMClient") as mock:
        client_instance = Mock()
        client_instance.model_name = "test-model"
        client_instance.complete.return_value = "Generated text"
        client_instance.complete_prompt.return_value = "Prompted text"
        mock.return_value = client_instance
        yield mock, client_instance


def test_create_inspectai_model_returns_model_adapter():
    """create_inspectai_model returns a MlflowInspectAIModel instance."""
    with patch("mlflow.genai.scorers.inspect_ai.models.ScorerLLMClient"):
        model = create_inspectai_model("openai:/gpt-4")
        assert isinstance(model, MlflowInspectAIModel)


def test_inspectai_model_initializes_backend_with_uri():
    """ScorerLLMClient is initialized with the exact model URI provided."""
    with patch("mlflow.genai.scorers.inspect_ai.models.ScorerLLMClient") as mock_client:
        create_inspectai_model("databricks:/my-endpoint")
        mock_client.assert_called_once_with("databricks:/my-endpoint")


def test_inspectai_model_complete_prompt_delegates_to_backend(mock_scorer_llm_client):
    """complete_prompt delegates to ScorerLLMClient.complete_prompt."""
    mock_client_class, mock_client_instance = mock_scorer_llm_client
    model = create_inspectai_model("openai:/gpt-4")
    result = model.complete_prompt("test prompt")
    assert result == "Prompted text"
    mock_client_instance.complete_prompt.assert_called_once_with("test prompt")


def test_inspectai_model_complete_delegates_to_backend(mock_scorer_llm_client):
    """complete delegates to ScorerLLMClient.complete with the message list."""
    mock_client_class, mock_client_instance = mock_scorer_llm_client
    model = create_inspectai_model("openai:/gpt-4")
    messages = [{"role": "user", "content": "test"}]
    result = model.complete(messages)
    assert result == "Generated text"
    mock_client_instance.complete.assert_called_once_with(messages)


def test_inspectai_model_passes_model_kwargs(mock_scorer_llm_client):
    """model_kwargs are stored on the adapter and accessible via model.model_kwargs."""
    mock_client_class, mock_client_instance = mock_scorer_llm_client
    model = create_inspectai_model(
        "openai:/gpt-4", model_kwargs={"temperature": 0.2, "max_tokens": 512}
    )
    assert model.model_kwargs == {"temperature": 0.2, "max_tokens": 512}


def test_inspectai_model_forwards_kwargs_to_complete_prompt(mock_scorer_llm_client):
    """model_kwargs are merged with per-call kwargs and forwarded to the backend."""
    mock_client_class, mock_client_instance = mock_scorer_llm_client
    model = create_inspectai_model(
        "openai:/gpt-4", model_kwargs={"temperature": 0.0}
    )
    model.complete_prompt("test prompt", max_tokens=256)
    mock_client_instance.complete_prompt.assert_called_once_with(
        "test prompt", temperature=0.0, max_tokens=256
    )


def test_inspectai_model_kwargs_override_model_defaults(mock_scorer_llm_client):
    """Per-call kwargs override model-level defaults when both are provided."""
    mock_client_class, mock_client_instance = mock_scorer_llm_client
    model = create_inspectai_model(
        "openai:/gpt-4", model_kwargs={"temperature": 0.2}
    )
    model.complete_prompt("test", temperature=0.8)
    mock_client_instance.complete_prompt.assert_called_once_with(
        "test", temperature=0.8
    )


def test_inspectai_model_name_property(mock_scorer_llm_client):
    """model_name property is delegated to the underlying ScorerLLMClient."""
    mock_client_class, mock_client_instance = mock_scorer_llm_client
    model = create_inspectai_model("openai:/gpt-4")
    assert model.model_name == "test-model"


def test_inspectai_model_empty_kwargs_default(mock_scorer_llm_client):
    """model_kwargs defaults to an empty dict when not provided."""
    mock_client_class, mock_client_instance = mock_scorer_llm_client
    model = create_inspectai_model("databricks")
    assert model.model_kwargs == {}


def test_create_inspectai_model_does_not_mutate_input_dict():
    """create_inspectai_model must not mutate the caller's model_kwargs dict."""
    kwargs = {"temperature": 0.5, "max_tokens": 100}
    with patch("mlflow.genai.scorers.inspect_ai.models.ScorerLLMClient"):
        create_inspectai_model("openai:/gpt-4", model_kwargs=kwargs)
    assert "temperature" in kwargs
