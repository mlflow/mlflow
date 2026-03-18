import importlib
from unittest.mock import Mock, patch

import pytest
import trulens  # noqa: F401

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.trulens.models import create_trulens_provider
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.trulens.models.call_chat_completions") as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


def test_create_trulens_provider_databricks(mock_call_chat_completions):
    mock_endpoint = Mock()
    mock_llm_provider = Mock()

    with patch.dict(
        "sys.modules",
        {
            "trulens.core.feedback.endpoint": mock_endpoint,
            "trulens.feedback.llm_provider": mock_llm_provider,
        },
    ):
        mock_llm_provider.LLMProvider = type(
            "LLMProvider", (), {"__init__": lambda self, **kw: None}
        )
        mock_endpoint.Endpoint = Mock()

        provider = create_trulens_provider("databricks")
        assert provider is not None

        # Call the provider to verify call_chat_completions is used
        provider._create_chat_completion(prompt="test prompt")
        mock_call_chat_completions.assert_called_once_with(
            user_prompt="test prompt",
            system_prompt="",
        )


def test_create_trulens_provider_databricks_endpoint_uses_litellm():
    mock_litellm_class = Mock()
    mock_litellm_class.return_value = Mock()

    with patch.dict("sys.modules", {"trulens.providers.litellm": Mock(LiteLLM=mock_litellm_class)}):
        from mlflow.genai.scorers.trulens import models

        importlib.reload(models)

        models.create_trulens_provider("databricks:/my-endpoint")
        mock_litellm_class.assert_called_once_with(model_engine="databricks/my-endpoint")


def test_create_trulens_provider_openai():
    mock_litellm_class = Mock()
    mock_litellm_class.return_value = Mock()

    with patch.dict("sys.modules", {"trulens.providers.litellm": Mock(LiteLLM=mock_litellm_class)}):
        from mlflow.genai.scorers.trulens import models

        importlib.reload(models)

        models.create_trulens_provider("openai:/gpt-4")
        mock_litellm_class.assert_called_once_with(model_engine="openai/gpt-4")


def test_create_trulens_provider_litellm_format():
    mock_litellm_class = Mock()
    mock_litellm_class.return_value = Mock()

    with patch.dict("sys.modules", {"trulens.providers.litellm": Mock(LiteLLM=mock_litellm_class)}):
        from mlflow.genai.scorers.trulens import models

        importlib.reload(models)

        models.create_trulens_provider("litellm:/gpt-4")
        mock_litellm_class.assert_called_once_with(model_engine="gpt-4")


def test_create_trulens_provider_invalid_format():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_trulens_provider("gpt-4")


def test_create_trulens_provider_gateway():
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    mock_litellm_class = Mock()
    mock_litellm_class.return_value = Mock()

    with patch.dict("sys.modules", {"trulens.providers.litellm": Mock(LiteLLM=mock_litellm_class)}):
        from mlflow.genai.scorers.trulens import models

        importlib.reload(models)

        with patch.object(
            models, "get_gateway_litellm_config", return_value=mock_config
        ) as mock_get_config:
            models.create_trulens_provider("gateway:/my-endpoint")

    mock_get_config.assert_called_once_with("my-endpoint")
    mock_litellm_class.assert_called_once_with(
        model_engine="openai/my-endpoint",
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
    )
