"""Tests for the SAP AI Core Orchestration v2 provider."""

from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.sap_ai_core import SapAiCoreAdapter, SapAiCoreConfig, SapAiCoreProvider


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


# ---------------------------------------------------------------------------
# SapAiCoreAdapter — request transformation
# ---------------------------------------------------------------------------


class TestSapAiCoreAdapterChatToModel:
    def test_basic_user_message(self):
        config = _make_endpoint_config()
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        result = SapAiCoreAdapter.chat_to_model(payload, config)

        assert result["config"]["modules"]["prompt_templating"]["prompt"]["template"] == [
            {"role": "user", "content": "Hello"}
        ]
        assert result["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o-mini"
        assert result["placeholder_values"] == {}

    def test_system_and_user_messages(self):
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

    def test_model_name_from_config(self):
        config = _make_endpoint_config(model_name="gpt-4o")
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        result = SapAiCoreAdapter.chat_to_model(payload, config)

        assert result["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o"

    def test_inference_params_forwarded_to_model_params(self):
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

    def test_no_params_block_when_no_inference_params(self):
        config = _make_endpoint_config()
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        result = SapAiCoreAdapter.chat_to_model(payload, config)
        model_block = result["config"]["modules"]["prompt_templating"]["model"]

        assert "params" not in model_block

    def test_unknown_payload_keys_not_forwarded(self):
        config = _make_endpoint_config()
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "response_format": {"type": "json_schema"},
        }
        result = SapAiCoreAdapter.chat_to_model(payload, config)

        assert "stream" not in result
        assert "response_format" not in result
        assert "params" not in result["config"]["modules"]["prompt_templating"]["model"]

    def test_placeholder_values_forwarded(self):
        config = _make_endpoint_config()
        placeholder = {"user_name": "Alice"}
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "placeholder_values": placeholder,
        }
        result = SapAiCoreAdapter.chat_to_model(payload, config)

        assert result["placeholder_values"] == placeholder

    def test_message_with_none_content_becomes_empty_string(self):
        config = _make_endpoint_config()
        payload = {"messages": [{"role": "assistant", "content": None}]}
        result = SapAiCoreAdapter.chat_to_model(payload, config)
        template = result["config"]["modules"]["prompt_templating"]["prompt"]["template"]

        assert template[0]["content"] == ""


# ---------------------------------------------------------------------------
# SapAiCoreAdapter — response decoding
# ---------------------------------------------------------------------------


class TestSapAiCoreAdapterModelToChat:
    def _make_orch_response(self, content: str = "Hello!") -> dict:
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
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return {
            "request_id": "ff3cb38e-215f-9c58-bb7f-45ee667d2d35",
            "intermediate_results": {"templating": [], "llm": inner},
            "final_result": inner,
        }

    def test_decodes_final_result(self):
        config = _make_endpoint_config()
        resp = self._make_orch_response("Hello from AI Core!")
        result = SapAiCoreAdapter.model_to_chat(resp, config)

        assert result.choices[0].message.content == "Hello from AI Core!"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_falls_back_to_root_when_no_final_result(self):
        """If for any reason final_result is absent, fall back to the root object."""
        config = _make_endpoint_config()
        bare_openai_resp = {
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "created": 1783313910,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Bare response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = SapAiCoreAdapter.model_to_chat(bare_openai_resp, config)

        assert result.choices[0].message.content == "Bare response"

    def test_request_id_preserved(self):
        config = _make_endpoint_config()
        resp = self._make_orch_response()
        result = SapAiCoreAdapter.model_to_chat(resp, config)

        assert result.id == "chatcmpl-abc123"


# ---------------------------------------------------------------------------
# SapAiCoreProvider
# ---------------------------------------------------------------------------


class TestSapAiCoreProvider:
    def test_adapter_class(self):
        provider = _make_provider()
        assert provider.adapter_class is SapAiCoreAdapter

    def test_headers_are_empty(self):
        provider = _make_provider()
        assert provider.headers == {}

    def test_display_name(self):
        assert SapAiCoreProvider.DISPLAY_NAME == "SAP AI Core"

    def test_config_type(self):
        assert SapAiCoreProvider.CONFIG_TYPE is SapAiCoreConfig

    def test_config_requires_no_api_key(self):
        # SapAiCoreConfig should be constructable with no arguments
        config = SapAiCoreConfig()
        assert config is not None

    def test_get_endpoint_url_reads_env(self, monkeypatch):
        monkeypatch.setenv(
            "MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL",
            "http://egress-gw.cluster.local/v2/inference/deployments/abc/chat/completions",
        )
        provider = _make_provider()
        url = provider.get_endpoint_url("llm/v1/chat")

        assert url == "http://egress-gw.cluster.local/v2/inference/deployments/abc/chat/completions"

    def test_get_endpoint_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(
            "MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL",
            "http://egress-gw.cluster.local/v2/chat/",
        )
        provider = _make_provider()
        assert not provider.get_endpoint_url("llm/v1/chat").endswith("/")

    def test_get_endpoint_url_raises_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL", raising=False)
        provider = _make_provider()
        with pytest.raises(MlflowException, match="MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL"):
            provider.get_endpoint_url("llm/v1/chat")

    def test_http_scheme_accepted(self, monkeypatch):
        monkeypatch.setenv(
            "MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL", "http://internal-gw/completions"
        )
        provider = _make_provider()
        url = provider.get_endpoint_url("llm/v1/chat")
        assert url.startswith("http://")

    def test_get_endpoint_url_raises_on_invalid_scheme(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL", "ftp://bad-scheme/completions")
        provider = _make_provider()
        with pytest.raises(MlflowException, match="http://"):
            provider.get_endpoint_url("llm/v1/chat")

    def test_https_scheme_accepted(self, monkeypatch):
        monkeypatch.setenv(
            "MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL", "https://public-api.example.com/completions"
        )
        provider = _make_provider()
        url = provider.get_endpoint_url("llm/v1/chat")
        assert url.startswith("https://")


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
# GatewayAdapter integration — is_applicable + request pipeline
# ---------------------------------------------------------------------------


def test_gateway_adapter_is_applicable_for_sap_ai_core():
    from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
    from mlflow.types.llm import ChatMessage

    assert GatewayAdapter.is_applicable(
        "sap-ai-core:/gpt-4o-mini", [ChatMessage(role="user", content="hello")]
    )


def test_gateway_adapter_sends_v2_body_and_extra_headers(monkeypatch):
    """End-to-end: verify POST body is Orchestration v2 format and extra_headers are forwarded."""
    monkeypatch.setenv(
        "MLFLOW_SAP_AI_CORE_ORCHESTRATION_URL",
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

    endpoint = call_kwargs["endpoint"]
    headers = call_kwargs["headers"]
    body = call_kwargs["payload"]

    assert endpoint == "http://egress-gw.cluster.local/v2/chat/completions"
    assert headers.get("AI-Resource-Group") == "default"
    assert headers.get("x-tenant-id") == "tenant-1"
    assert "Authorization" not in headers

    assert "config" in body
    assert (
        body["config"]["modules"]["prompt_templating"]["model"]["name"] == "gpt-4o-mini"
    )
    template = body["config"]["modules"]["prompt_templating"]["prompt"]["template"]
    assert template[0]["role"] == "user"
    assert template[0]["content"] == "Is this safe?"

    assert result.feedback.value in ("yes", "no")
