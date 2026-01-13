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


def test_create_trulens_provider_databricks_endpoint(mock_invoke_serving_endpoint):
    provider = create_trulens_provider("databricks:/my-endpoint")
    assert provider is not None
    assert hasattr(provider, "_create_chat_completion")


def test_create_trulens_provider_openai(monkeypatch):
    from trulens.providers.openai import OpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = create_trulens_provider("openai:/gpt-4")
    assert isinstance(provider, OpenAI)
    assert provider.model_engine == "gpt-4"


def test_create_trulens_provider_invalid_format():
    with pytest.raises(MlflowException, match="Invalid model_uri format"):
        create_trulens_provider("gpt-4")
