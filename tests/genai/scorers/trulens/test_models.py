from unittest.mock import Mock, patch

import pytest
import trulens  # noqa: F401

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.trulens.models import create_trulens_provider


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.trulens.models.call_chat_completions") as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_invoke_serving_endpoint():
    with patch("mlflow.genai.scorers.trulens.models._invoke_databricks_serving_endpoint") as mock:
        result = Mock()
        result.response = "Endpoint output"
        mock.return_value = result
        yield mock


def test_create_trulens_provider_databricks(mock_call_chat_completions):
    provider = create_trulens_provider("databricks")
    assert provider is not None
    assert hasattr(provider, "_create_chat_completion")

    # Test that _create_chat_completion calls the underlying method
    result = provider._create_chat_completion(prompt="Test prompt")
    assert result == "Test output"
    mock_call_chat_completions.assert_called_once()


def test_create_trulens_provider_databricks_endpoint(mock_invoke_serving_endpoint):
    provider = create_trulens_provider("databricks:/my-endpoint")
    assert provider is not None
    assert hasattr(provider, "_create_chat_completion")

    # Test that _create_chat_completion calls the underlying method
    result = provider._create_chat_completion(prompt="Test prompt")
    assert result == "Endpoint output"
    mock_invoke_serving_endpoint.assert_called_once()


def test_create_trulens_provider_openai(monkeypatch):
    from trulens.providers.litellm import LiteLLM

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    try:
        provider = create_trulens_provider("openai:/gpt-4")
        assert isinstance(provider, LiteLLM)
        assert provider.model_engine == "openai/gpt-4"
    except AttributeError as e:
        # TruLens LiteLLM provider has an instrumentation bug with CallTypes enum
        if "CallTypes" in str(e):
            pytest.skip("TruLens LiteLLM instrumentation bug - see TruLens issue tracker")
        raise


def test_create_trulens_provider_invalid_format():
    with pytest.raises(MlflowException, match="Invalid model_uri format"):
        create_trulens_provider("gpt-4")
