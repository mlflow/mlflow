"""Gateway-based judge invocation with tool-calling loop.

Gateway-based judge invocation that makes sync HTTP calls to LLM providers
via the gateway infrastructure.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic
import requests

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_GATEWAY_CALLER_HEADER, GatewayCaller
from mlflow.genai.judges.types import JudgeMessage
from mlflow.genai.judges.utils.tool_calling_utils import (
    _process_tool_calls,
    _raise_iteration_limit_exceeded,
)
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

_logger = logging.getLogger(__name__)

# Global cache to track model capabilities across function calls
_MODEL_RESPONSE_FORMAT_CAPABILITIES: dict[str, bool] = {}

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Default timeout for HTTP requests to LLM providers (seconds)
_REQUEST_TIMEOUT = 60


@dataclass
class InvokeOutput:
    response: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None


def _get_default_judge_response_schema() -> type[pydantic.BaseModel]:
    from mlflow.genai.judges.base import Judge

    output_fields = Judge.get_output_fields()
    field_definitions = {}
    for field in output_fields:
        field_definitions[field.name] = (str, pydantic.Field(description=field.description))
    return pydantic.create_model("JudgeEvaluation", **field_definitions)


def _build_request(
    model: str,
    messages: list[JudgeMessage],
    tools: list[dict[str, Any]] | None,
    response_format: type[pydantic.BaseModel] | None,
    include_response_format: bool,
    inference_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the OpenAI-format chat completions request payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [msg.to_dict() for msg in messages],
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    if include_response_format:
        schema_cls = response_format or _get_default_judge_response_schema()
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema_cls.model_json_schema(),
                "strict": True,
            },
        }

    if inference_params:
        payload.update(inference_params)

    return payload


def _send_chat_request(
    api_base: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    num_retries: int,
) -> dict[str, Any]:
    """Send a chat completions request with retry logic."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    last_exception = None
    for attempt in range(1 + num_retries):
        try:
            resp = requests.post(
                url=url,
                headers={"Content-Type": "application/json", **headers},
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )

            if resp.status_code == 400:
                # Don't retry 400s — raise immediately for caller to handle
                error_body = _safe_parse_error(resp)
                raise ChatCompletionError(
                    status_code=400,
                    message=error_body,
                    is_context_window_error=_is_context_window_error(error_body),
                )

            if resp.status_code in _RETRYABLE_STATUS_CODES:
                last_exception = ChatCompletionError(
                    status_code=resp.status_code,
                    message=_safe_parse_error(resp),
                )
                if attempt < num_retries:
                    _sleep_with_backoff(attempt)
                    continue
                raise last_exception

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt < num_retries:
                _sleep_with_backoff(attempt)
                continue
            raise MlflowException(
                f"Request to {url} timed out after {_REQUEST_TIMEOUT}s",
                error_code=INTERNAL_ERROR,
            ) from e
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            if attempt < num_retries:
                _sleep_with_backoff(attempt)
                continue
            raise MlflowException(
                f"Failed to connect to {url}: {e}",
                error_code=INTERNAL_ERROR,
            ) from e

    raise last_exception


class ChatCompletionError(Exception):
    def __init__(self, status_code: int, message: str, is_context_window_error: bool = False):
        self.status_code = status_code
        self.message = message
        self.is_context_window_error = is_context_window_error
        super().__init__(message)


def _safe_parse_error(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if "error" in body:
            error = body["error"]
            if isinstance(error, dict):
                return error.get("message", resp.text)
            return str(error)
    except Exception:
        pass
    return resp.text


def _is_context_window_error(error_message: str) -> bool:
    lower = error_message.lower()
    return "context length" in lower or "too many tokens" in lower or "maximum context" in lower


def _sleep_with_backoff(attempt: int) -> None:
    delay = min(2**attempt, 30)
    time.sleep(delay)


def _resolve_provider_config(
    provider: str,
    model_name: str,
    base_url: str | None,
    extra_headers: dict[str, str] | None,
) -> tuple[str, str, dict[str, str]]:
    """Resolve API base URL, model identifier, and headers for a provider.

    Returns:
        (api_base, model, headers)
    """
    if provider == "gateway":
        from mlflow.genai.utils.gateway_utils import get_gateway_litellm_config

        config = get_gateway_litellm_config(model_name)
        headers = {**(config.extra_headers or {})}
        headers[MLFLOW_GATEWAY_CALLER_HEADER] = GatewayCaller.JUDGE.value
        # For direct HTTP calls, we use the endpoint name as the model.
        return config.api_base, model_name, headers

    # Direct provider — resolve API key from environment
    from mlflow.utils.providers import _CORE_PROVIDER_ENV_VARS

    headers = {}
    if extra_headers:
        headers.update(extra_headers)

    env_var_config = _CORE_PROVIDER_ENV_VARS.get(provider)
    if isinstance(env_var_config, str):
        api_key = os.environ.get(env_var_config)
        if api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"

    api_base = base_url or _get_default_api_base(provider)

    return api_base, model_name, headers


# Providers that use OpenAI-compatible chat/completions API format.
# Anthropic and Gemini use different API formats and must go through
# their respective provider adapter classes for single-shot calls.
_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "mistral", "gateway"}


def _get_default_api_base(provider: str) -> str:
    defaults = {
        "openai": "https://api.openai.com/v1",
        "mistral": "https://api.mistral.ai/v1",
    }
    if provider in defaults:
        return defaults[provider]
    raise MlflowException(
        f"No default API base URL for provider '{provider}'. Please provide a base_url parameter.",
        error_code=INTERNAL_ERROR,
    )


def _get_max_context_tokens(provider: str, model: str) -> int | None:
    """Look up the max input token limit for a model from the vendored model prices JSON."""
    from mlflow.utils.providers import _get_model_cost

    model_cost = _get_model_cost()
    # Try provider/model format first (e.g., "openai/gpt-4.1")
    for key in (f"{provider}/{model}", model):
        if key in model_cost:
            return model_cost[key].get("max_input_tokens")
    return None


def _should_proactively_prune(
    usage: dict[str, Any],
    max_context_tokens: int | None,
    threshold: float = 0.85,
) -> bool:
    """Check if prompt token usage is approaching the context window limit.

    Uses the actual prompt_tokens count from the LLM response to decide
    whether to proactively prune the conversation history before the next call.
    """
    if max_context_tokens is None:
        return False
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        return False
    return prompt_tokens > max_context_tokens * threshold


def _parse_response_message(response_data: dict[str, Any]) -> JudgeMessage:
    """Parse the assistant message from an OpenAI-format chat completions response."""
    choices = response_data.get("choices", [])
    if not choices:
        raise MlflowException("Empty choices in chat completions response")

    message_data = choices[0].get("message", {})
    return JudgeMessage(
        role=message_data.get("role", "assistant"),
        content=message_data.get("content"),
        tool_calls=message_data.get("tool_calls"),
    )


def _remove_oldest_tool_call_pair(
    messages: list[JudgeMessage],
) -> list[JudgeMessage] | None:
    """Remove the oldest assistant message with tool calls and its corresponding tool responses."""
    result = next(
        ((i, msg) for i, msg in enumerate(messages) if msg.role == "assistant" and msg.tool_calls),
        None,
    )
    if result is None:
        return None

    assistant_idx, assistant_msg = result
    modified = messages[:]
    modified.pop(assistant_idx)

    tool_call_ids = {tc["id"] for tc in assistant_msg.tool_calls}
    return [
        msg for msg in modified if not (msg.role == "tool" and msg.tool_call_id in tool_call_ids)
    ]


def invoke_via_gateway_and_handle_tools(
    provider: str,
    model_name: str,
    messages: list["ChatMessage"],
    trace: "Trace | None",
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None = None,
    inference_params: dict[str, Any] | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> InvokeOutput:
    """
    Invoke an LLM with tool-calling loop support.

    Uses sync HTTP calls to the appropriate provider endpoint. Handles the
    iterative tool-calling loop until the LLM produces a final answer.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic', 'gateway').
        model_name: The model name (or endpoint name for gateway provider).
        messages: List of ChatMessage objects.
        trace: Optional trace object for context with tool calling support.
        num_retries: Number of retries with exponential backoff on transient failures.
        response_format: Optional Pydantic model class for structured output format.
        inference_params: Optional dictionary of additional inference parameters.
        base_url: Optional base URL to route requests through.
        extra_headers: Optional dictionary of additional HTTP headers.

    Returns:
        InvokeOutput with the model's response and usage metadata.

    Raises:
        MlflowException: If the request fails after all retries, or if the provider
            does not support OpenAI-compatible chat completions format.
    """
    from mlflow.genai.judges.tools import list_judge_tools
    from mlflow.protos.databricks_pb2 import BAD_REQUEST

    # When a custom base_url is provided, we assume it's an OpenAI-compatible endpoint.
    # Otherwise, only providers known to use OpenAI format are supported.
    if provider not in _OPENAI_COMPATIBLE_PROVIDERS and base_url is None:
        raise MlflowException(
            f"Provider '{provider}' does not support the OpenAI-compatible chat completions "
            f"format required for trace-based tool calling. Supported providers for "
            f"trace-based judging: {', '.join(sorted(_OPENAI_COMPATIBLE_PROVIDERS))}. "
            f"You can also pass a base_url pointing to an OpenAI-compatible endpoint.",
            error_code=BAD_REQUEST,
        )

    judge_messages = [JudgeMessage(role=msg.role, content=msg.content) for msg in messages]

    api_base, model, headers = _resolve_provider_config(
        provider,
        model_name,
        base_url,
        extra_headers,
    )

    tools = []
    if trace is not None:
        judge_tools = list_judge_tools()
        tools = [tool.get_definition().to_dict() for tool in judge_tools]

    include_response_format = _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(f"{provider}/{model}", True)

    max_context_tokens = _get_max_context_tokens(provider, model)
    max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            _raise_iteration_limit_exceeded(max_iterations)

        try:
            payload = _build_request(
                model=model,
                messages=judge_messages,
                tools=tools or None,
                response_format=response_format,
                include_response_format=include_response_format,
                inference_params=inference_params,
            )

            try:
                response_data = _send_chat_request(
                    api_base=api_base,
                    headers=headers,
                    payload=payload,
                    num_retries=num_retries,
                )
            except ChatCompletionError as e:
                if e.is_context_window_error:
                    pruned = _remove_oldest_tool_call_pair(judge_messages)
                    if pruned is None:
                        raise MlflowException(
                            "Context window exceeded and there are no tool calls to truncate. "
                            "The initial prompt may be too long for the model's context window."
                        ) from e
                    judge_messages = pruned
                    continue

                if e.status_code == 400 and include_response_format:
                    _logger.debug(
                        f"Model {provider}/{model} may not support structured outputs. "
                        f"Error: {e.message}. Falling back to unstructured response.",
                    )
                    _MODEL_RESPONSE_FORMAT_CAPABILITIES[f"{provider}/{model}"] = False
                    include_response_format = False
                    continue

                raise MlflowException(
                    f"Failed to invoke judge model: {e.message}",
                    error_code=INTERNAL_ERROR,
                ) from e

            message = _parse_response_message(response_data)

            if not message.tool_calls:
                usage = response_data.get("usage", {})
                return InvokeOutput(
                    response=message.content,
                    request_id=response_data.get("id"),
                    num_prompt_tokens=usage.get("prompt_tokens"),
                    num_completion_tokens=usage.get("completion_tokens"),
                )

            judge_messages.append(message)
            tool_response_messages = _process_tool_calls(tool_calls=message.tool_calls, trace=trace)
            judge_messages.extend(tool_response_messages)

            # Proactively prune if approaching context window limit,
            # using the actual prompt_tokens from the LLM response.
            usage = response_data.get("usage", {})
            if _should_proactively_prune(usage, max_context_tokens):
                pruned = _remove_oldest_tool_call_pair(judge_messages)
                if pruned is not None:
                    _logger.debug(
                        f"Proactively pruned conversation history "
                        f"(prompt_tokens={usage.get('prompt_tokens')}, "
                        f"max={max_context_tokens})"
                    )
                    judge_messages = pruned

        except MlflowException:
            raise
        except Exception as e:
            raise MlflowException(
                f"Failed to invoke the judge model: {e}",
                error_code=INTERNAL_ERROR,
            ) from e
