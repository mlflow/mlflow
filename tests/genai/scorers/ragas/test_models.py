import sys
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.models import MlflowRagasLLM, create_ragas_model


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
    with patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
    ) as mock:
        result = Mock()
        result.output = '{"answer": "Test output", "score": 42}'
        mock.return_value = result
        yield mock


def test_databricks_ragas_llm_generate(mock_call_chat_completions):
    from mlflow.genai.scorers.llm_backend import ScorerLLMClient

    backend = ScorerLLMClient("databricks")
    llm = MlflowRagasLLM(backend)
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
        model="databricks",
    )


def test_create_ragas_model_databricks():
    model = create_ragas_model("databricks")
    assert model.__class__.__name__ == "MlflowRagasLLM"


def test_create_ragas_model_databricks_serving_endpoint():
    model = create_ragas_model("databricks:/my-endpoint")
    assert model.__class__.__name__ == "MlflowRagasLLM"
    assert model.get_model_name() == "databricks/my-endpoint"


def test_create_ragas_model_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = create_ragas_model("openai:/gpt-4")
    assert model.__class__.__name__ == "MlflowRagasLLM"
    assert model.get_model_name() == "openai/gpt-4"


def test_create_ragas_model_rejects_provider_no_slash():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_ragas_model("openai:gpt-4")


def test_create_ragas_model_rejects_model_name_only():
    with pytest.raises(MlflowException, match="Malformed model uri"):
        create_ragas_model("gpt-4")


def test_create_ragas_model_gateway_uses_native_provider():
    from mlflow.genai.scorers.ragas.models import MlflowRagasLLM

    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_get_provider:
        model = create_ragas_model("gateway:/my-endpoint")
    mock_get_provider.assert_called_once()

    assert isinstance(model, MlflowRagasLLM)
    assert model.get_model_name() == "gateway/my-endpoint"


@pytest.mark.parametrize("provider", ["cohere", "mosaicml", "palm"])
def test_create_ragas_model_registered_but_unsupported_falls_back_to_litellm(provider):
    model = create_ragas_model(f"{provider}:/my-model")
    assert model.__class__.__name__ == "LiteLLMStructuredLLM"
