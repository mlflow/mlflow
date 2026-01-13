from unittest.mock import Mock, patch

import phoenix.evals as phoenix_evals
import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.phoenix.models import (
    DatabricksPhoenixModel,
    DatabricksServingEndpointPhoenixModel,
    create_phoenix_model,
)


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.phoenix.models.call_chat_completions") as mock:
        result = Mock()
        result.output = "Test output"
        mock.return_value = result
        yield mock


@pytest.fixture
def mock_invoke_serving_endpoint():
    with patch("mlflow.genai.scorers.phoenix.models._invoke_databricks_serving_endpoint") as mock:
        result = Mock()
        result.response = "Endpoint output"
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


def test_databricks_serving_endpoint_model_call(mock_invoke_serving_endpoint):
    model = DatabricksServingEndpointPhoenixModel("my-endpoint")
    result = model("Test prompt")

    assert result == "Endpoint output"
    mock_invoke_serving_endpoint.assert_called_once_with(
        model_name="my-endpoint",
        prompt="Test prompt",
        num_retries=3,
        response_format=None,
    )


def test_databricks_serving_endpoint_model_get_model_name():
    model = DatabricksServingEndpointPhoenixModel("my-endpoint")
    assert model.get_model_name() == "databricks:/my-endpoint"


def test_create_phoenix_model_databricks():
    model = create_phoenix_model("databricks")
    assert isinstance(model, DatabricksPhoenixModel)
    assert model.get_model_name() == "databricks"


def test_create_phoenix_model_databricks_endpoint():
    model = create_phoenix_model("databricks:/my-endpoint")
    assert isinstance(model, DatabricksServingEndpointPhoenixModel)
    assert model.get_model_name() == "databricks:/my-endpoint"


def test_create_phoenix_model_openai():
    with patch(
        "litellm.validate_environment",
        return_value={"keys_in_environment": True, "missing_keys": []},
    ):
        model = create_phoenix_model("openai:/gpt-4")
        assert isinstance(model, phoenix_evals.LiteLLMModel)


def test_create_phoenix_model_invalid_format():
    with pytest.raises(MlflowException, match="Invalid model_uri format"):
        create_phoenix_model("gpt-4")
