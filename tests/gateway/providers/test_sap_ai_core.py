from typing import Any
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.sap_ai_core import (
    SapAiCoreAdapter,
    SapAiCoreConfig,
    SapAiCoreProvider,
)


def _make_endpoint_config(model_name: str = "gpt-4o-mini") -> EndpointConfig:
    return EndpointConfig(
        name="sap-ai-core-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "sap-ai-core",
            "name": model_name,
            "config": {},
        },
    )


def _make_provider(model_name: str = "gpt-4o-mini") -> SapAiCoreProvider:
    return SapAiCoreProvider(_make_endpoint_config(model_name))


def _make_orch_response(content: str = "Hello!") -> dict[str, Any]:
    inner = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1783313910,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return {
        "request_id": "ff3cb38e-215f-9c58-bb7f-45ee667d2d35",
        "intermediate_results": {"templating": [], "llm": inner},
        "final_result": inner,
    }


# ---------------------------------------------------------------------------
# SapAiCoreAdapter — request transformation
# ---------------------------------------------------------------------------


def test_adapter_basic_user_message():
    config = _make_endpoint_config()
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert result["config"]["modules"]["prompt_templating"]["prompt"]["template"] == [
        {"role": "user", "content": "Hello"}
    ]
    assert result["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o-mini"
    assert result["placeholder_values"] == {}


def test_adapter_system_and_user_messages():
    config = _make_endpoint_config()
    payload = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is MLflow?"},
        ]
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assert len(template) == 2
    assert template[0] == {"role": "system", "content": "You are helpful."}
    assert template[1] == {"role": "user", "content": "What is MLflow?"}


def test_adapter_model_name_from_config():
    config = _make_endpoint_config(model_name="gpt-4o")
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert result["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o"


def test_adapter_inference_params_forwarded():
    config = _make_endpoint_config()
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 300,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

    assert params["max_tokens"] == 300
    assert params["temperature"] == 0.5
    assert params["top_p"] == 0.9


def test_adapter_response_format_forwarded():
    config = _make_endpoint_config()
    response_format = {"type": "json_schema", "json_schema": {"name": "Out", "strict": True}}
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": response_format,
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

    assert params["response_format"] == response_format


def test_adapter_tools_forwarded():
    config = _make_endpoint_config()
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "tool_choice": "auto",
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    params = result["config"]["modules"]["prompt_templating"]["model"]["params"]

    assert params["tools"] == tools
    assert params["tool_choice"] == "auto"


def test_adapter_no_params_block_when_no_inference_params():
    config = _make_endpoint_config()
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    model_block = result["config"]["modules"]["prompt_templating"]["model"]

    assert "params" not in model_block


def test_adapter_unknown_keys_not_forwarded():
    config = _make_endpoint_config()
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert "stream" not in result
    assert "params" not in result["config"]["modules"]["prompt_templating"]["model"]


def test_adapter_placeholder_values_forwarded():
    config = _make_endpoint_config()
    placeholder = {"user_name": "Alice"}
    payload = {"messages": [{"role": "user", "content": "hi"}], "placeholder_values": placeholder}
    result = SapAiCoreAdapter.chat_to_model(payload, config)

    assert result["placeholder_values"] == placeholder


def test_adapter_none_content_becomes_empty_string():
    config = _make_endpoint_config()
    payload = {"messages": [{"role": "assistant", "content": None}]}
    result = SapAiCoreAdapter.chat_to_model(payload, config)
    template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

    assert template[0]["content"] == ""


# ---------------------------------------------------------------------------
# SapAiCoreAdapter — response decoding
# ---------------------------------------------------------------------------


def test_adapter_decodes_final_result():
    config = _make_endpoint_config()
    result = SapAiCoreAdapter.model_to_chat(_make_orch_response("Hello from AI Core!"), config)

    assert result.choices[0].message.content == "Hello from AI Core!"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5


def test_adapter_falls_back_to_root_when_no_final_result():
    config = _make_endpoint_config()
    bare = {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1783313910,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Bare"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    result = SapAiCoreAdapter.model_to_chat(bare, config)

    assert result.choices[0].message.content == "Bare"


def test_adapter_request_id_preserved():
    config = _make_endpoint_config()
    result = SapAiCoreAdapter.model_to_chat(_make_orch_response(), config)

    assert result.id == "chatcmpl-abc123"


# ---------------------------------------------------------------------------
# SapAiCoreProvider
# ---------------------------------------------------------------------------


def test_provider_adapter_class():
    assert _make_provider().adapter_class is SapAiCoreAdapter


def test_provider_headers_are_empty():
    assert _make_provider().headers == {}


def test_provider_display_name():
    assert SapAiCoreProvider.DISPLAY_NAME == "SAP AI Core"


def test_provider_config_type():
    assert SapAiCoreProvider.CONFIG_TYPE is SapAiCoreConfig


def test_provider_config_requires_no_api_key():
    assert SapAiCoreConfig() is not None


def test_provider_get_endpoint_url_reads_env(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_GENAI_JUDGE_BASE_URL",
        "http://egress-gw.cluster.local/v2/inference/deployments/abc/chat/completions",
    )
    url = _make_provider().get_endpoint_url("llm/v1/chat")

    assert url == "http://egress-gw.cluster.local/v2/inference/deployments/abc/chat/completions"


def test_provider_get_endpoint_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "http://egress-gw.cluster.local/v2/chat/")
    assert not _make_provider().get_endpoint_url("llm/v1/chat").endswith("/")


def test_provider_get_endpoint_url_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_GENAI_JUDGE_BASE_URL", raising=False)
    with pytest.raises(MlflowException, match="MLFLOW_GENAI_JUDGE_BASE_URL"):
        _make_provider().get_endpoint_url("llm/v1/chat")


def test_provider_get_endpoint_url_raises_on_invalid_scheme(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "ftp://bad-scheme/completions")
    with pytest.raises(MlflowException, match="http://"):
        _make_provider().get_endpoint_url("llm/v1/chat")


def test_provider_http_scheme_accepted(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "http://internal-gw/completions")
    assert _make_provider().get_endpoint_url("llm/v1/chat").startswith("http://")


def test_provider_https_scheme_accepted(monkeypatch):
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_BASE_URL", "https://public-api.example.com/completions")
    assert _make_provider().get_endpoint_url("llm/v1/chat").startswith("https://")


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


def test_sap_ai_core_is_supported_provider():
    from mlflow.gateway.provider_registry import is_supported_provider

    assert is_supported_provider("sap-ai-core")


def test_sap_ai_core_in_provider_enum():
    from mlflow.gateway.config import Provider

    assert Provider.SAP_AI_CORE == "sap-ai-core"


# ---------------------------------------------------------------------------
# GatewayAdapter integration
# ---------------------------------------------------------------------------


def test_gateway_adapter_is_applicable_for_sap_ai_core():
    from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
    from mlflow.types.llm import ChatMessage

    assert GatewayAdapter.is_applicable(
        "sap-ai-core:/gpt-4o-mini", [ChatMessage(role="user", content="hello")]
    )


def test_gateway_adapter_sends_v2_body_and_extra_headers(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_GENAI_JUDGE_BASE_URL",
        "http://egress-gw.cluster.local/v2/chat/completions",
    )

    final_result = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1783313910,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"result": "yes", "rationale": "looks good"}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    fake_response = {
        "request_id": "req-123",
        "intermediate_results": {},
        "final_result": final_result,
    }

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.send_chat_request",
        return_value=fake_response,
    ) as mock_send:
        from mlflow.genai.judges.adapters.base_adapter import AdapterInvocationInput
        from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
        from mlflow.types.llm import ChatMessage

        adapter = GatewayAdapter()
        params = AdapterInvocationInput(
            model_uri="sap-ai-core:/gpt-4o-mini",
            prompt=[ChatMessage(role="user", content="Is this safe?")],
            assessment_name="safety",
            extra_headers={"AI-Resource-Group": "default", "x-tenant-id": "tenant-1"},
        )
        result = adapter._invoke(params)

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs

    assert call_kwargs["endpoint"] == "http://egress-gw.cluster.local/v2/chat/completions"
    assert call_kwargs["headers"].get("AI-Resource-Group") == "default"
    assert call_kwargs["headers"].get("x-tenant-id") == "tenant-1"
    assert "Authorization" not in call_kwargs["headers"]
    assert "config" in call_kwargs["payload"]
    assert (
        call_kwargs["payload"]["config"]["modules"]["prompt_templating"]["model"]["name"]
        == "gpt-4o-mini"
    )
    assert result.feedback.value in ("yes", "no")
