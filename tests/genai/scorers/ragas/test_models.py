import functools
import sys
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.models import DatabricksRagasLLM, create_ragas_model
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig


class DummyResponseModel(BaseModel):
    answer: str
    score: int


@pytest.fixture(autouse=True)
def _mock_litellm_module():
    if "litellm" not in sys.modules:
        mock = Mock()
        mock.acompletion = Mock()
        with patch.dict(sys.modules, {"litellm": mock}):
            yield mock
    else:
        yield


@pytest.fixture
def mock_call_chat_completions():
    with patch("mlflow.genai.scorers.ragas.models.call_chat_completions") as mock:
        result = Mock()
        result.output = '{"answer": "Test output", "score": 42}'
        mock.return_value = result
        yield mock


def test_databricks_ragas_llm_generate(mock_call_chat_completions):
    llm = DatabricksRagasLLM()
    result = llm.generate(prompt="Test prompt", response_model=DummyResponseModel)

    assert isinstance(result, DummyResponseModel)
    assert result.answer == "Test output"
    assert result.score == 42
    mock_call_chat_completions.assert_called_once_with(
        user_prompt=(
            "Test prompt\n\nOUTPUT FORMAT: Respond ONLY with a JSON object "
            'containing these fields: "answer", "score", no other text. '
            "Do not add markdown formatting to the response."
        ),
        system_prompt="",
    )


def test_create_ragas_model_databricks():
    model = create_ragas_model("databricks")
    assert model.__class__.__name__ == "DatabricksRagasLLM"


def test_create_ragas_model_databricks_serving_endpoint():
    model = create_ragas_model("databricks:/my-endpoint")
    assert model.__class__.__name__ == "LiteLLMStructuredLLM"


def test_create_ragas_model_openai():
    model = create_ragas_model("openai:/gpt-4")
    assert model.__class__.__name__ == "LiteLLMStructuredLLM"


def test_create_ragas_model_rejects_provider_no_slash():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_ragas_model("openai:gpt-4")


def test_create_ragas_model_rejects_model_name_only():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_ragas_model("gpt-4")


def test_create_ragas_model_gateway():
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    with patch(
        "mlflow.genai.scorers.ragas.models.get_gateway_litellm_config",
        return_value=mock_config,
    ) as mock_get_config:
        model = create_ragas_model("gateway:/my-endpoint")

    mock_get_config.assert_called_once_with("my-endpoint")
    assert model.__class__.__name__ == "LiteLLMStructuredLLM"
    assert model.model == "openai/my-endpoint"


def test_create_ragas_model_gateway_uses_partial_with_api_base_and_key():
    mock_config = GatewayLiteLLMConfig(
        api_base="http://localhost:5000/gateway/mlflow/v1/",
        api_key="mlflow-gateway-auth",
        model="openai/my-endpoint",
        extra_headers=None,
    )
    mock_litellm = Mock()
    with (
        patch.dict(sys.modules, {"litellm": mock_litellm}),
        patch(
            "mlflow.genai.scorers.ragas.models.get_gateway_litellm_config",
            return_value=mock_config,
        ),
        patch("mlflow.genai.scorers.ragas.models.instructor") as mock_instructor,
    ):
        mock_instructor.from_litellm.return_value = Mock()
        create_ragas_model("gateway:/my-endpoint")

    mock_instructor.from_litellm.assert_called_once()
    partial_arg = mock_instructor.from_litellm.call_args[0][0]
    assert isinstance(partial_arg, functools.partial)
    assert partial_arg.keywords["api_base"] == "http://localhost:5000/gateway/mlflow/v1/"
    assert partial_arg.keywords["api_key"] == "mlflow-gateway-auth"
    assert partial_arg.func is mock_litellm.acompletion
