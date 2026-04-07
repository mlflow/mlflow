from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic
import requests

import mlflow
from mlflow.gateway.config import EndpointType
from mlflow.genai.judges.adapters.litellm_adapter import _is_litellm_available
from mlflow.metrics.genai.model_utils import _get_provider_instance, _parse_model_uri, _send_request
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking._tracking_service.utils import _get_store

if TYPE_CHECKING:
    import litellm

    from mlflow.gateway.schemas.chat import ResponsePayload

_logger = logging.getLogger(__name__)

# Defaults for LLM calls; callers may override via inference_params.
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_NUM_RETRIES = 5


def _resolve_model_for_gateway(endpoint_name: str) -> str | None:
    """
    Resolve a gateway model name to its actual provider/model URI.
    """
    try:
        endpoint = _get_store().get_gateway_endpoint(name=endpoint_name)
        if endpoint and endpoint.model_mappings:
            m = endpoint.model_mappings[0]
            if model_def := m.model_definition:
                return f"{model_def.provider}:/{model_def.model_name}"
    except Exception:
        _logger.debug(
            "Failed to resolve gateway model %r for cost lookup", endpoint_name, exc_info=True
        )


class _TokenCounter:
    """Thread-safe accumulator for LLM token usage across pipeline phases."""

    def __init__(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ):
        self._lock = threading.RLock()
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self._cost_usd = cost_usd
        self._cost_resolved = False
        provider, model_name = _parse_model_uri(model)
        if provider == "gateway":
            self._model = _resolve_model_for_gateway(model_name)
        else:
            self._model = model

    @property
    def cost_usd(self) -> float | None:
        """Return the total cost, falling back to the LiteLLM pricing API if needed."""
        with self._lock:
            if self._cost_usd == 0 and not self._cost_resolved:
                total = self.input_tokens + self.output_tokens
                if total > 0 and self._model:
                    if cost := _lookup_model_cost(
                        self._model, self.input_tokens, self.output_tokens
                    ):
                        self._cost_usd = cost
                # Mark resolved once we've attempted lookup with tokens present,
                # or when there are no tokens yet (nothing to look up).
                self._cost_resolved = total > 0 or not self._model
            return self._cost_usd or None

    def add_cost(self, cost: float) -> None:
        with self._lock:
            self._cost_usd += cost

    def track(self, response: litellm.ModelResponse | ResponsePayload) -> None:
        with self._lock:
            if response.usage:
                self.input_tokens += response.usage.prompt_tokens or 0
                self.output_tokens += response.usage.completion_tokens or 0
            if hidden := getattr(response, "_hidden_params", None):
                if cost := hidden.get("response_cost"):
                    self.add_cost(cost)

    def to_dict(self) -> dict[str, int | float]:
        result = {}
        total = self.input_tokens + self.output_tokens
        if total > 0:
            result["input_tokens"] = self.input_tokens
            result["output_tokens"] = self.output_tokens
            result["total_tokens"] = total
        if cost := self.cost_usd:
            result["cost_usd"] = round(cost, 6)
        return result


@trace_disabled
def _call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    response_format: type[pydantic.BaseModel] | None = None,
    token_counter: _TokenCounter | None = None,
    inference_params: dict[str, Any] | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    num_retries: int = _DEFAULT_NUM_RETRIES,
) -> Any:
    if _is_litellm_available():
        return _call_llm_via_litellm(
            model,
            messages,
            json_mode=json_mode,
            response_format=response_format,
            token_counter=token_counter,
            inference_params=inference_params,
            max_tokens=max_tokens,
            num_retries=num_retries,
        )
    return _call_llm_via_gateway(
        model,
        messages,
        json_mode=json_mode,
        response_format=response_format,
        token_counter=token_counter,
        inference_params=inference_params,
        max_tokens=max_tokens,
        num_retries=num_retries,
    )


def _call_llm_via_litellm(
    model: str,
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    response_format: type[pydantic.BaseModel] | None = None,
    token_counter: _TokenCounter | None = None,
    inference_params: dict[str, Any] | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    num_retries: int = _DEFAULT_NUM_RETRIES,
) -> Any:
    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm
    from mlflow.genai.utils.gateway_utils import get_gateway_litellm_config
    from mlflow.metrics.genai.model_utils import _parse_model_uri, convert_mlflow_uri_to_litellm

    provider, model_name = _parse_model_uri(model)

    if provider == "gateway":
        config = get_gateway_litellm_config(model_name)
        litellm_model = config.model
        api_base = config.api_base
        api_key = config.api_key
        extra_headers = config.extra_headers
    else:
        litellm_model = convert_mlflow_uri_to_litellm(model)
        api_base = None
        api_key = None
        extra_headers = None

    use_format = response_format or ({"type": "json_object"} if json_mode else None)
    merged_params = {"max_completion_tokens": max_tokens}
    if inference_params:
        merged_params.update(inference_params)
    response = _invoke_litellm(
        litellm_model=litellm_model,
        messages=messages,
        tools=[],
        num_retries=num_retries,
        response_format=use_format,
        include_response_format=use_format is not None,
        inference_params=merged_params,
        api_base=api_base,
        api_key=api_key,
        extra_headers=extra_headers,
    )
    if token_counter is not None:
        token_counter.track(response)
    return response


def _call_llm_via_gateway(
    model: str,
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    response_format: type[pydantic.BaseModel] | None = None,
    token_counter: _TokenCounter | None = None,
    inference_params: dict[str, Any] | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    num_retries: int = _DEFAULT_NUM_RETRIES,
) -> Any:
    # Lightweight fallback for when LiteLLM is not installed. Supports
    # providers with MLflow gateway adapters (OpenAI, Anthropic, Gemini, Mistral)
    # and the MLflow AI Gateway (gateway:/ URIs).
    # Known gaps vs the LiteLLM path: no drop_params
    # (https://docs.litellm.ai/docs/completion/drop_params) - LiteLLM silently
    # strips unsupported params (e.g. response_format) per model before sending
    # the request, while this path sends them as-is. Not an issue for OpenAI
    # and Anthropic which both support structured outputs. Also missing:
    # no context window management and no per-request cost tracking.
    provider_name, model_name = _parse_model_uri(model)
    provider = _get_provider_instance(provider_name, model_name)

    payload = {"messages": messages, "max_completion_tokens": max_tokens}
    if inference_params:
        payload.update(inference_params)
    if response_format is not None:
        payload["response_format"] = _pydantic_to_response_format(response_format)
    elif json_mode:
        payload["response_format"] = {"type": "json_object"}

    chat_payload = provider.adapter_class.chat_to_model(payload, provider.config)

    for attempt in range(num_retries + 1):
        try:
            raw_response = _send_request(
                endpoint=provider.get_endpoint_url(EndpointType.LLM_V1_CHAT),
                headers=provider.headers,
                payload=chat_payload,
            )
            break
        except (
            requests.exceptions.RequestException,
            mlflow.exceptions.MlflowException,
        ):
            if attempt >= num_retries:
                raise
            time.sleep(2**attempt)

    response = provider.adapter_class.model_to_chat(raw_response, provider.config)
    if token_counter is not None:
        token_counter.track(response)
    return response


def _pydantic_to_response_format(cls: type[pydantic.BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": cls.__name__,
            "schema": cls.model_json_schema(),
        },
    }


@dataclass(frozen=True)
class _ModelCost:
    input_cost_per_token: float
    output_cost_per_token: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _ModelCost:
        return cls(
            input_cost_per_token=data.get("input_cost_per_token") or 0,
            output_cost_per_token=data.get("output_cost_per_token") or 0,
        )


@functools.lru_cache(maxsize=64)
def _fetch_model_cost(provider: str, model_name: str) -> _ModelCost | None:
    from mlflow.utils.providers import _lookup_model_info

    if info := _lookup_model_info(model_name, custom_llm_provider=provider):
        return _ModelCost.from_dict(info)
    return None


def _lookup_model_cost(model_uri: str, input_tokens: int, output_tokens: int) -> float | None:
    provider, model_name = _parse_model_uri(model_uri)
    if cost := _fetch_model_cost(provider, model_name):
        return input_tokens * cost.input_cost_per_token + output_tokens * cost.output_cost_per_token
    return None
