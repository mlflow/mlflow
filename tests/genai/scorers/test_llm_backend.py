from unittest.mock import Mock, patch

import pydantic
import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.llm_backend import ScorerLLMClient


def test_databricks_route():
    backend = ScorerLLMClient("databricks")
    assert backend.route == "databricks"
    assert backend.is_native
    assert backend.model_name == "databricks"
    assert backend.provider is None


def test_databricks_agentic_route():
    backend = ScorerLLMClient("gpt-oss-120b")
    assert backend.route == "databricks"
    assert backend.is_native
    assert backend.model_name == "gpt-oss-120b"


def test_endpoints_route():
    backend = ScorerLLMClient("endpoints:/my-endpoint")
    assert backend.route == "endpoints"
    assert backend.is_native
    assert backend.provider == "endpoints"
    assert backend.model_name == "endpoints/my-endpoint"


def test_native_route(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = ScorerLLMClient("openai:/gpt-4")
    assert backend.route == "native"
    assert backend.is_native
    assert backend.provider == "openai"
    assert backend.model_name == "openai/gpt-4"
    assert backend.raw_model_name == "gpt-4"


def test_litellm_fallback_route():
    backend = ScorerLLMClient("some_unknown:/model")
    assert backend.route == "litellm"
    assert not backend.is_native
    assert backend.provider == "some_unknown"
    assert backend.model_name == "some_unknown/model"


def test_known_provider_missing_config_falls_back_to_litellm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = ScorerLLMClient("openai:/gpt-4")
    assert backend.route == "litellm"


def test_gateway_route():
    with patch("mlflow.genai.scorers.llm_backend._get_provider_instance") as mock_provider:
        backend = ScorerLLMClient("gateway:/my-endpoint")
    mock_provider.assert_called_once()
    assert backend.route == "native"
    assert backend.is_native
    assert backend.provider == "gateway"


def test_complete_databricks():
    with patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
    ) as mock_call:
        result = Mock()
        result.output = "Hello from databricks"
        mock_call.return_value = result

        backend = ScorerLLMClient("databricks")
        response = backend.complete([{"role": "user", "content": "Hi"}])

    assert response == "Hello from databricks"
    mock_call.assert_called_once()


def test_complete_native(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value="native response",
    ) as mock_call:
        backend = ScorerLLMClient("openai:/gpt-4")
        response = backend.complete([{"role": "user", "content": "Hi"}])

    assert response == "native response"
    mock_call.assert_called_once_with(
        "openai",
        "gpt-4",
        messages=[{"role": "user", "content": "Hi"}],
        eval_parameters=None,
        response_format=None,
    )


def test_complete_prompt_convenience(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value="response",
    ) as mock_call:
        backend = ScorerLLMClient("openai:/gpt-4")
        response = backend.complete_prompt("Hello")

    assert response == "response"
    mock_call.assert_called_once_with(
        "openai",
        "gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        eval_parameters=None,
        response_format=None,
    )


def test_complete_with_pydantic_response_format(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class TestSchema(pydantic.BaseModel):
        result: str

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value='{"result": "yes"}',
    ) as mock_call:
        backend = ScorerLLMClient("openai:/gpt-4")
        backend.complete(
            [{"role": "user", "content": "test"}],
            response_format=TestSchema,
        )

    mock_call.assert_called_once()
    call_kwargs = mock_call.call_args
    assert call_kwargs.kwargs["response_format"] is not None
    assert isinstance(call_kwargs.kwargs["response_format"], dict)


def test_complete_with_dict_response_format(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pre_converted = {"type": "json_schema", "json_schema": {"name": "test"}}

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        return_value="response",
    ) as mock_call:
        backend = ScorerLLMClient("openai:/gpt-4")
        backend.complete(
            [{"role": "user", "content": "test"}],
            response_format=pre_converted,
        )

    mock_call.assert_called_once()
    assert mock_call.call_args.kwargs["response_format"] is pre_converted


def test_complete_with_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    call_count = 0

    def mock_dispatch_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise MlflowException("transient error")
        return "success"

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        side_effect=mock_dispatch_side_effect,
    ):
        backend = ScorerLLMClient("openai:/gpt-4")
        response = backend.complete(
            [{"role": "user", "content": "test"}],
            num_retries=1,
        )

    assert response == "success"
    assert call_count == 2


def test_complete_retry_exhausted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "mlflow.genai.scorers.llm_backend._call_llm_provider_api",
        side_effect=MlflowException("persistent error"),
    ):
        backend = ScorerLLMClient("openai:/gpt-4")

        with pytest.raises(MlflowException, match="persistent error"):
            backend.complete(
                [{"role": "user", "content": "test"}],
                num_retries=1,
            )


def test_complete_litellm_fallback():
    mock_litellm = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "litellm response"
    mock_litellm.completion.return_value = mock_response

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        backend = ScorerLLMClient("some_unknown:/model")
        response = backend.complete([{"role": "user", "content": "test"}])

    assert response == "litellm response"
    mock_litellm.completion.assert_called_once()


def test_complete_litellm_not_installed():
    with patch.dict("sys.modules", {"litellm": None}):
        backend = ScorerLLMClient("some_unknown:/model")
        with pytest.raises(MlflowException, match="not natively supported"):
            backend.complete([{"role": "user", "content": "test"}])


def test_complete_databricks_extracts_system_prompt():
    with patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
    ) as mock_call:
        result = Mock()
        result.output = "response"
        mock_call.return_value = result

        backend = ScorerLLMClient("databricks")
        backend.complete([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ])

    mock_call.assert_called_once_with(
        user_prompt="Hi", system_prompt="You are helpful", model="databricks"
    )
