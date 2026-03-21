from unittest.mock import Mock, patch

import phoenix.evals as phoenix_evals
import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.phoenix.models import (
    DatabricksPhoenixModel,
    create_phoenix_model,
)
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.phoenix.models.call_chat_completions") as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


def test_databricks_phoenix_model_call(mock_call_chat_completions):
    model = DatabricksPhoenixModel()
    result = model("Test prompt")

    assert result == "Test output"
    mock_call_chat_completions.assert_called_once_with(
        user_prompt="Test prompt",
        system_prompt="",
    )


def test_databricks_phoenix_model_get_model_name():
    model = DatabricksPhoenixModel()
    assert model.get_model_name() == "databricks"


def test_create_phoenix_model_databricks():
    model = create_phoenix_model("databricks")
    assert isinstance(model, DatabricksPhoenixModel)
    assert model.get_model_name() == "databricks"


def test_create_phoenix_model_databricks_endpoint():
    model = create_phoenix_model("databricks:/my-endpoint")
    assert isinstance(model, phoenix_evals.LiteLLMModel)
    assert model.model == "databricks/my-endpoint"


def test_create_phoenix_model_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = create_phoenix_model("openai:/gpt-4")
    assert isinstance(model, phoenix_evals.LiteLLMModel)


def test_create_phoenix_model_invalid_format():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_phoenix_model("gpt-4")


def test_create_phoenix_model_gateway(monkeypatch):
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "mlflow-gateway-auth")
    with patch(
        "mlflow.genai.scorers.phoenix.models.get_gateway_litellm_config",
        return_value=mock_config,
    ) as mock_get_config:
        model = create_phoenix_model("gateway:/my-endpoint")

    mock_get_config.assert_called_once_with("my-endpoint")
    assert isinstance(model, phoenix_evals.LiteLLMModel)
    assert model.model == "openai/my-endpoint"
    assert model.model_kwargs["api_base"] == "http://localhost:5000/gateway/mlflow/v1/"
    assert model.model_kwargs["api_key"] == "mlflow-gateway-auth"


def test_create_phoenix_model_gateway_sets_api_base_and_key():
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    mock_litellm_model = Mock()
    with (
        patch(
            "mlflow.genai.scorers.phoenix.models.get_gateway_litellm_config",
            return_value=mock_config,
        ),
        patch("phoenix.evals.LiteLLMModel", mock_litellm_model),
    ):
        create_phoenix_model("gateway:/my-endpoint")

    mock_litellm_model.assert_called_once_with(
        model="openai/my-endpoint",
        model_kwargs={
            "api_base": "http://localhost:5000/gateway/mlflow/v1/",
            "api_key": "mlflow-gateway-auth",
            "drop_params": True,
        },
    )
