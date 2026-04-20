from unittest import mock

import pydantic
import pytest

from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig
from mlflow.genai.utils.llm_utils import (
    _call_llm,
    _call_llm_via_gateway,
    _fetch_model_cost,
    _lookup_model_cost,
    _ModelCost,
    _pydantic_to_response_format,
    _resolve_model_for_gateway,
    _TokenCounter,
)
from mlflow.types.chat import ChatChoice, ChatCompletionResponse, ChatMessage, ChatUsage


def test_token_counter_tracks_usage():
    counter = _TokenCounter(model="openai:/gpt-5-mini")
    assert counter.input_tokens == 0
    assert counter.output_tokens == 0
    assert counter.cost_usd is None

    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response._hidden_params = {"response_cost": 0.005}

    counter.track(mock_response)

    assert counter.input_tokens == 100
    assert counter.output_tokens == 50
    assert counter.cost_usd == 0.005


def test_token_counter_tracks_gateway_response_without_hidden_params():
    counter = _TokenCounter(model="openai:/gpt-5-mini")
    response = ChatCompletionResponse(
        created=0,
        model="gpt-5-mini",
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content="hi"))],
        usage=ChatUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280),
    )

    counter.track(response)

    assert counter.input_tokens == 200
    assert counter.output_tokens == 80
    assert counter._cost_usd == 0.0
    assert counter._model == "openai:/gpt-5-mini"


def test_token_counter_to_dict_looks_up_cost_when_zero():
    counter = _TokenCounter(input_tokens=100, output_tokens=50, model="openai:/gpt-5-mini")

    with mock.patch(
        "mlflow.genai.utils.llm_utils._lookup_model_cost",
        return_value=0.0042,
    ) as mock_lookup:
        result = counter.to_dict()

    mock_lookup.assert_called_once()
    assert result["cost_usd"] == 0.0042
    assert result["total_tokens"] == 150


def test_call_llm_uses_gateway_when_litellm_unavailable():
    with (
        mock.patch(
            "mlflow.genai.utils.llm_utils._is_litellm_available", return_value=False
        ) as mock_avail,
        mock.patch(
            "mlflow.genai.utils.llm_utils._call_llm_via_gateway",
        ) as mock_gw,
    ):
        _call_llm("openai:/gpt-5-mini", [{"role": "user", "content": "hi"}])

    mock_avail.assert_called_once()
    mock_gw.assert_called_once()


def test_call_llm_uses_litellm_when_available():
    with (
        mock.patch(
            "mlflow.genai.utils.llm_utils._is_litellm_available", return_value=True
        ) as mock_avail,
        mock.patch(
            "mlflow.genai.utils.llm_utils._call_llm_via_litellm",
        ) as mock_ll,
    ):
        _call_llm("openai:/gpt-5-mini", [{"role": "user", "content": "hi"}])

    mock_avail.assert_called_once()
    mock_ll.assert_called_once()


def test_pydantic_to_response_format():
    class MySchema(pydantic.BaseModel):
        name: str
        score: int

    result = _pydantic_to_response_format(MySchema)

    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "MySchema"
    schema = result["json_schema"]["schema"]
    assert "name" in schema["properties"]
    assert "score" in schema["properties"]


def test_lookup_model_cost_returns_calculated_cost():
    cost_info = _ModelCost(input_cost_per_token=0.00001, output_cost_per_token=0.00003)
    with mock.patch(
        "mlflow.genai.utils.llm_utils._fetch_model_cost", return_value=cost_info
    ) as mock_fetch:
        cost = _lookup_model_cost("openai:/gpt-5-mini", 1000, 500)

    mock_fetch.assert_called_once()
    assert cost == pytest.approx(1000 * 0.00001 + 500 * 0.00003)


def test_lookup_model_cost_returns_none_on_missing_model():
    with mock.patch(
        "mlflow.genai.utils.llm_utils._fetch_model_cost", return_value=None
    ) as mock_fetch:
        assert _lookup_model_cost("openai:/gpt-5-mini", 100, 50) is None

    mock_fetch.assert_called_once()


def test_call_llm_handles_gateway_models():
    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    gateway_config = GatewayLiteLLMConfig(
        model="openai/test-endpoint",
        api_base="http://localhost:5000/gateway",
        api_key="test-key",
        extra_headers={"X-Custom": "header"},
    )

    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=True),
        mock.patch(
            "mlflow.genai.utils.gateway_utils.get_gateway_litellm_config",
            return_value=gateway_config,
        ) as mock_get_config,
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ) as mock_invoke,
    ):
        messages = [{"role": "user", "content": "test"}]
        result = _call_llm("gateway:/test-endpoint", messages)

        mock_get_config.assert_called_once_with("test-endpoint")
        mock_invoke.assert_called_once()

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["litellm_model"] == "openai/test-endpoint"
        assert call_kwargs["api_base"] == "http://localhost:5000/gateway"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["extra_headers"] == {"X-Custom": "header"}
        assert call_kwargs["messages"] == messages
        assert result == mock_response


def test_call_llm_handles_non_gateway_models():
    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=True),
        mock.patch(
            "mlflow.metrics.genai.model_utils.convert_mlflow_uri_to_litellm",
            return_value="openai/gpt-4",
        ) as mock_convert,
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ) as mock_invoke,
    ):
        messages = [{"role": "user", "content": "test"}]
        result = _call_llm("openai:/gpt-4", messages)

        mock_convert.assert_called_once_with("openai:/gpt-4")
        mock_invoke.assert_called_once()

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["litellm_model"] == "openai/gpt-4"
        assert call_kwargs["api_base"] is None
        assert call_kwargs["api_key"] is None
        assert call_kwargs["extra_headers"] is None
        assert call_kwargs["messages"] == messages
        assert result == mock_response


def test_call_llm_with_json_mode():
    mock_response = mock.MagicMock()
    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=True),
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ) as mock_invoke,
    ):
        messages = [{"role": "user", "content": "test"}]
        _call_llm("openai:/gpt-4", messages, json_mode=True)

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["include_response_format"] is True


def test_call_llm_with_response_format():
    class TestModel(pydantic.BaseModel):
        field: str

    mock_response = mock.MagicMock()
    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=True),
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ) as mock_invoke,
    ):
        messages = [{"role": "user", "content": "test"}]
        _call_llm("openai:/gpt-4", messages, response_format=TestModel)

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["response_format"] == TestModel
        assert call_kwargs["include_response_format"] is True


def test_call_llm_tracks_tokens():
    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response._hidden_params = {"response_cost": 0.01}

    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=True),
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ),
    ):
        counter = _TokenCounter(model="openai:/gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        _call_llm("openai:/gpt-4", messages, token_counter=counter)

        assert counter.input_tokens == 100
        assert counter.output_tokens == 50
        assert counter.cost_usd == 0.01


def test_call_llm_inference_params_forwarded_to_litellm():
    mock_response = mock.MagicMock()
    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=True),
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ) as mock_invoke,
    ):
        messages = [{"role": "user", "content": "test"}]
        _call_llm(
            "openai:/gpt-4",
            messages,
            inference_params={"temperature": 0.5, "max_completion_tokens": 1024},
        )

        call_kwargs = mock_invoke.call_args[1]
        # inference_params should override the default max_completion_tokens
        assert call_kwargs["inference_params"]["temperature"] == 0.5
        assert call_kwargs["inference_params"]["max_completion_tokens"] == 1024


def test_call_llm_inference_params_forwarded_to_gateway():
    mock_provider = mock.MagicMock()
    captured_payload = {}
    mock_provider.adapter_class.chat_to_model.side_effect = lambda payload, config: (
        captured_payload.update(payload) or payload
    )
    mock_provider.get_endpoint_url.return_value = "http://localhost:5000/v1/chat/completions"
    mock_provider.headers = {}

    mock_chat_response = mock.MagicMock()
    mock_chat_response.usage = mock.MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_provider.adapter_class.model_to_chat.return_value = mock_chat_response

    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=False),
        mock.patch(
            "mlflow.genai.utils.llm_utils._get_provider_instance",
            return_value=mock_provider,
        ),
        mock.patch(
            "mlflow.genai.utils.llm_utils._send_request",
            return_value={},
        ),
    ):
        _call_llm(
            "openai:/gpt-4",
            [{"role": "user", "content": "test"}],
            inference_params={"temperature": 0.7, "max_completion_tokens": 512},
        )

    assert captured_payload["temperature"] == 0.7
    # inference_params should override the default max_completion_tokens
    assert captured_payload["max_completion_tokens"] == 512


# ---- gateway:/ URI via _get_provider_instance ----


def test_call_llm_via_gateway_dispatches_gateway_uri_without_litellm():
    mock_provider = mock.MagicMock()
    mock_provider.adapter_class.chat_to_model.side_effect = lambda payload, config: payload
    mock_provider.get_endpoint_url.return_value = (
        "http://localhost:5000/gateway/mlflow/v1/chat/completions"
    )
    mock_provider.headers = {"Authorization": "Bearer token"}

    raw_response = {
        "id": "chatcmpl-gw",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "my-endpoint",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "gateway response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    mock_chat_response = mock.MagicMock()
    mock_chat_response.choices = [
        mock.MagicMock(message=mock.MagicMock(content="gateway response"))
    ]
    mock_chat_response.usage = mock.MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_provider.adapter_class.model_to_chat.return_value = mock_chat_response

    with (
        mock.patch("mlflow.genai.utils.llm_utils._is_litellm_available", return_value=False),
        mock.patch(
            "mlflow.genai.utils.llm_utils._get_provider_instance",
            return_value=mock_provider,
        ) as mock_get_provider,
        mock.patch(
            "mlflow.genai.utils.llm_utils._send_request",
            return_value=raw_response,
        ) as mock_send,
    ):
        messages = [{"role": "user", "content": "test"}]
        result = _call_llm("gateway:/my-endpoint", messages)

    mock_get_provider.assert_called_once_with("gateway", "my-endpoint")
    mock_send.assert_called_once()
    assert result.choices[0].message.content == "gateway response"


def _make_mapping(provider=None, model_name=None):
    mapping = mock.MagicMock()
    if provider:
        mapping.model_definition.provider = provider
        mapping.model_definition.model_name = model_name
    else:
        mapping.model_definition = None
    return mapping


def _make_store(model_mappings=(), raises=False):
    store = mock.MagicMock()
    if raises:
        store.get_gateway_endpoint.side_effect = Exception()
    else:
        endpoint = mock.MagicMock()
        endpoint.model_mappings = list(model_mappings)
        store.get_gateway_endpoint.return_value = endpoint
    return store


@pytest.mark.parametrize(
    ("mock_store", "expected"),
    [
        (_make_store([_make_mapping("openai", "gpt-4o")]), "openai:/gpt-4o"),
        (_make_store(), None),
        (_make_store([_make_mapping()]), None),
        (_make_store(raises=True), None),
    ],
    ids=["success", "no_mappings", "no_model_def", "exception"],
)
def test_resolve_model_for_gateway(mock_store, expected):
    with mock.patch("mlflow.genai.utils.llm_utils._get_store", return_value=mock_store):
        assert _resolve_model_for_gateway("my-endpoint") == expected


def test_token_counter_resolves_gateway_model_on_init():
    with mock.patch(
        "mlflow.genai.utils.llm_utils._resolve_model_for_gateway",
        return_value="openai:/gpt-4o",
    ) as mock_resolve:
        counter = _TokenCounter(model="gateway:/my-endpoint")

    mock_resolve.assert_called_once_with("my-endpoint")
    assert counter._model == "openai:/gpt-4o"


def test_token_counter_does_not_resolve_non_gateway_model():
    with mock.patch("mlflow.genai.utils.llm_utils._resolve_model_for_gateway") as mock_resolve:
        counter = _TokenCounter(model="openai:/gpt-4o")

    mock_resolve.assert_not_called()
    assert counter._model == "openai:/gpt-4o"


@pytest.mark.parametrize(
    ("model_info", "provider", "model_name", "expected_cost"),
    [
        (
            {"input_cost_per_token": 0.00001, "output_cost_per_token": 0.00003},
            "openai",
            "gpt-4o",
            _ModelCost(input_cost_per_token=0.00001, output_cost_per_token=0.00003),
        ),
        (None, "openai", "unknown-model", None),
    ],
)
def test_fetch_model_cost(model_info, provider, model_name, expected_cost):
    with mock.patch(
        "mlflow.utils.providers._lookup_model_info", return_value=model_info
    ) as mock_lookup:
        _fetch_model_cost.cache_clear()
        result = _fetch_model_cost(provider, model_name)

    mock_lookup.assert_called_once_with(model_name, custom_llm_provider=provider)
    assert result == expected_cost


def test_lookup_model_cost_passes_provider_and_model():
    cost_info = _ModelCost(input_cost_per_token=1, output_cost_per_token=3)
    with mock.patch(
        "mlflow.genai.utils.llm_utils._fetch_model_cost", return_value=cost_info
    ) as mock_fetch:
        cost = _lookup_model_cost("anthropic:/claude-3-5-sonnet", 1000, 500)

    mock_fetch.assert_called_once_with("anthropic", "claude-3-5-sonnet")
    assert cost == 1000 * 1 + 500 * 3


def test_call_llm_via_gateway_uses_resolved_model_from_token_counter():
    mock_provider = mock.MagicMock()
    mock_provider.adapter_class.model_to_chat.return_value.usage.prompt_tokens = 10
    mock_provider.adapter_class.model_to_chat.return_value.usage.completion_tokens = 5

    with mock.patch(
        "mlflow.genai.utils.llm_utils._resolve_model_for_gateway",
        return_value="openai:/gpt-4o",
    ):
        counter = _TokenCounter(model="gateway:/my-endpoint")

    assert counter._model == "openai:/gpt-4o"

    with (
        mock.patch(
            "mlflow.genai.utils.llm_utils._get_provider_instance",
            return_value=mock_provider,
        ),
        mock.patch("mlflow.genai.utils.llm_utils._send_request", return_value={}),
    ):
        _call_llm_via_gateway(
            "gateway:/my-endpoint", [{"role": "user", "content": "hi"}], token_counter=counter
        )

    assert counter._model == "openai:/gpt-4o"
