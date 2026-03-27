"""Gateway-based judge adapter with tool-calling loop support.

Makes sync HTTP calls to LLM providers via the gateway infrastructure,
with retry logic, context window management, and proactive pruning.
"""

from __future__ import annotations

import json
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

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointType
from mlflow.gateway.constants import MLFLOW_GATEWAY_CALLER_HEADER, GatewayCaller
from mlflow.genai.discovery.utils import _pydantic_to_response_format
from mlflow.genai.judges.adapters.base_adapter import (
    AdapterInvocationInput,
    AdapterInvocationOutput,
    BaseJudgeAdapter,
)
from mlflow.genai.judges.utils.parsing_utils import (
    _sanitize_justification,
    _strip_markdown_code_blocks,
)
from mlflow.genai.judges.utils.tool_calling_utils import (
    _process_tool_calls,
    _raise_iteration_limit_exceeded,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)

# "endpoints" is a special case for MLflow deployment endpoints (e.g. Databricks model serving).
_NATIVE_PROVIDERS = ["openai", "anthropic", "gemini", "mistral", "endpoints"]

# Providers that use OpenAI-compatible chat/completions API format.
# Anthropic and Gemini use different API formats and must go through
# their respective provider adapter classes for single-shot calls.
_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "mistral", "gateway"}

# Global cache to track model capabilities across function calls
_MODEL_RESPONSE_FORMAT_CAPABILITIES: dict[str, bool] = {}

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Default timeout for HTTP requests to LLM providers (seconds)
_REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InvokeOutput:
    response: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None


class ChatCompletionError(Exception):
    def __init__(self, status_code: int, message: str, is_context_window_error: bool = False):
        self.status_code = status_code
        self.message = message
        self.is_context_window_error = is_context_window_error
        super().__init__(message)


# ---------------------------------------------------------------------------
# HTTP / request helpers
# ---------------------------------------------------------------------------


def _message_to_dict(msg: "ChatMessage") -> dict[str, Any]:
    """Serialize a ChatMessage to OpenAI message format, omitting None fields."""
    d: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_calls is not None:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        d["name"] = msg.name
    return d


def _get_default_judge_response_schema() -> type[pydantic.BaseModel]:
    from mlflow.genai.judges.base import Judge

    output_fields = Judge.get_output_fields()
    field_definitions = {}
    for field in output_fields:
        field_definitions[field.name] = (str, pydantic.Field(description=field.description))
    return pydantic.create_model("JudgeEvaluation", **field_definitions)


def _build_request(
    model: str,
    messages: list["ChatMessage"],
    tools: list[dict[str, Any]] | None,
    response_format: type[pydantic.BaseModel] | None,
    include_response_format: bool,
    inference_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the OpenAI-format chat completions request payload."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_to_dict(msg) for msg in messages],
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


# ---------------------------------------------------------------------------
# Provider config resolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Context window management
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Response parsing and message management
# ---------------------------------------------------------------------------


def _parse_response_message(response_data: dict[str, Any]) -> "ChatMessage":
    """Parse the assistant message from an OpenAI-format chat completions response."""
    from mlflow.types.llm import ChatMessage

    choices = response_data.get("choices", [])
    if not choices:
        raise MlflowException("Empty choices in chat completions response")

    message_data = choices[0].get("message", {})

    # tool_calls come as plain dicts; ChatMessage.__post_init__ auto-converts to ToolCall
    tool_calls = message_data.get("tool_calls")
    content = message_data.get("content")

    # ChatMessage requires content when tool_calls is absent
    if content is None and not tool_calls:
        content = ""

    return ChatMessage(
        role=message_data.get("role", "assistant"),
        content=content,
        tool_calls=tool_calls,
    )


def _remove_oldest_tool_call_pair(
    messages: list["ChatMessage"],
) -> list["ChatMessage"] | None:
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

    tool_call_ids = {tc.id for tc in assistant_msg.tool_calls}
    return [
        msg for msg in modified if not (msg.role == "tool" and msg.tool_call_id in tool_call_ids)
    ]


# ---------------------------------------------------------------------------
# Single-shot gateway invocation (no tool calling)
# ---------------------------------------------------------------------------


def _invoke_via_gateway(
    model_uri: str,
    provider: str,
    prompt: str | list[dict[str, str]],
    inference_params: dict[str, Any] | None = None,
    response_format: type[pydantic.BaseModel] | None = None,
    base_url: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> str:
    """
    Invoke the judge model via native AI Gateway adapters.

    Supports both string prompts (via ``score_model_on_payload``) and
    ChatMessage-style message lists (via the provider infrastructure).
    """
    from mlflow.metrics.genai.model_utils import (
        _call_llm_provider_api,
        _parse_model_uri,
        get_endpoint_type,
        score_model_on_payload,
    )

    if provider not in _NATIVE_PROVIDERS:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=BAD_REQUEST,
        )

    if isinstance(prompt, str):
        return score_model_on_payload(
            model_uri=model_uri,
            payload=prompt,
            eval_parameters=inference_params,
            extra_headers=extra_headers,
            proxy_url=base_url,
            endpoint_type=get_endpoint_type(model_uri) or EndpointType.LLM_V1_CHAT,
        )

    _, model_name = _parse_model_uri(model_uri)
    rf_dict = _pydantic_to_response_format(response_format) if response_format else None
    return _call_llm_provider_api(
        provider,
        model_name,
        messages=prompt,
        eval_parameters=inference_params,
        response_format=rf_dict,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GatewayAdapter(BaseJudgeAdapter):
    """Adapter for native AI Gateway providers (fallback when LiteLLM is not available)."""

    @classmethod
    def is_applicable(
        cls,
        model_uri: str,
        prompt: str | list["ChatMessage"],
    ) -> bool:
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        model_provider, _ = _parse_model_uri(model_uri)
        if model_provider not in _NATIVE_PROVIDERS:
            return False
        # "endpoints" (Databricks model serving) only supports string prompts
        # via score_model_on_payload; _get_provider_instance doesn't handle it.
        if isinstance(prompt, list) and model_provider == "endpoints":
            return False
        return True

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        # base_url and extra_headers are not supported for deployment endpoints
        if input_params.model_provider == "endpoints" and (
            input_params.base_url is not None or input_params.extra_headers is not None
        ):
            raise MlflowException(
                "base_url and extra_headers are not supported for deployment "
                "endpoints (endpoints:/...). The endpoint URL is determined by the "
                "deployment target configuration.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # When a trace is provided, use the tool-calling loop
        if input_params.trace is not None:
            return self._invoke_with_tools(input_params)

        if isinstance(input_params.prompt, str):
            prompt = input_params.prompt
        else:
            prompt = [{"role": msg.role, "content": msg.content} for msg in input_params.prompt]

        response = _invoke_via_gateway(
            input_params.model_uri,
            input_params.model_provider,
            prompt,
            inference_params=input_params.inference_params,
            response_format=input_params.response_format,
            base_url=input_params.base_url,
            extra_headers=input_params.extra_headers,
        )

        cleaned_response = _strip_markdown_code_blocks(response)

        try:
            response_dict = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse response from judge model. Response: {response}",
                error_code=BAD_REQUEST,
            ) from e

        feedback = Feedback(
            name=input_params.assessment_name,
            value=response_dict["result"],
            rationale=_sanitize_justification(response_dict.get("rationale", "")),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=input_params.model_uri
            ),
        )

        return AdapterInvocationOutput(feedback=feedback)

    def invoke_with_structured_output(
        self,
        model_uri: str,
        messages: list["ChatMessage"],
        output_schema: type[pydantic.BaseModel],
        trace: "Trace | None" = None,
        num_retries: int = 10,
        inference_params: dict[str, Any] | None = None,
    ) -> pydantic.BaseModel:
        """Invoke the model and parse the response into a Pydantic schema."""
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        provider, model_name = _parse_model_uri(model_uri)

        output = self._run_tool_calling_loop(
            provider=provider,
            model_name=model_name,
            messages=messages,
            trace=trace,
            num_retries=num_retries,
            response_format=output_schema,
            inference_params=inference_params,
        )

        cleaned_response = _strip_markdown_code_blocks(output.response)
        response_dict = json.loads(cleaned_response)
        return output_schema(**response_dict)

    def _invoke_with_tools(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        """Invoke the judge model with trace-based tool calling support."""
        from mlflow.tracing.constant import AssessmentMetadataKey
        from mlflow.types.llm import ChatMessage

        messages = (
            [ChatMessage(role="user", content=input_params.prompt)]
            if isinstance(input_params.prompt, str)
            else input_params.prompt
        )

        output = self._run_tool_calling_loop(
            provider=input_params.model_provider,
            model_name=input_params.model_name,
            messages=messages,
            trace=input_params.trace,
            num_retries=input_params.num_retries,
            response_format=input_params.response_format,
            inference_params=input_params.inference_params,
            base_url=input_params.base_url,
            extra_headers=input_params.extra_headers,
        )

        cleaned_response = _strip_markdown_code_blocks(output.response)

        try:
            response_dict = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse response from judge model. Response: {output.response}",
                error_code=BAD_REQUEST,
            ) from e

        metadata = {}
        if output.num_prompt_tokens:
            metadata[AssessmentMetadataKey.JUDGE_INPUT_TOKENS] = output.num_prompt_tokens
        if output.num_completion_tokens:
            metadata[AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS] = output.num_completion_tokens
        metadata = metadata or None

        feedback = Feedback(
            name=input_params.assessment_name,
            value=response_dict["result"],
            rationale=_sanitize_justification(response_dict.get("rationale", "")),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id=input_params.model_uri
            ),
            trace_id=input_params.trace.info.trace_id if input_params.trace else None,
            metadata=metadata,
        )

        return AdapterInvocationOutput(feedback=feedback)

    def _run_tool_calling_loop(
        self,
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
        """Run the tool-calling loop with sync HTTP calls to the provider."""
        from mlflow.genai.judges.tools import list_judge_tools
        from mlflow.types.llm import ChatMessage

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

        judge_messages = [ChatMessage(role=msg.role, content=msg.content) for msg in messages]

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

        include_response_format = _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(
            f"{provider}/{model}", True
        )

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
                                "Context window exceeded and there are no tool calls to "
                                "truncate. The initial prompt may be too long for the "
                                "model's context window."
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
                tool_response_messages = _process_tool_calls(
                    tool_calls=message.tool_calls, trace=trace
                )
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
