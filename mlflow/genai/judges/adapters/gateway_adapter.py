"""Gateway-based judge adapter with tool-calling loop support.

Uses the MLflow Gateway provider infrastructure for request/response
transformation and provider configuration, with retry logic, context
window management, and proactive pruning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic

from mlflow.utils.providers import _lookup_model_info

if TYPE_CHECKING:
    from mlflow.entities.trace import Trace
    from mlflow.gateway.providers.base import BaseProvider
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointType
from mlflow.gateway.constants import MLFLOW_GATEWAY_CALLER_HEADER, GatewayCaller
from mlflow.gateway.provider_registry import is_supported_provider
from mlflow.genai.judges.adapters.base_adapter import (
    AdapterInvocationInput,
    AdapterInvocationOutput,
    BaseJudgeAdapter,
)
from mlflow.genai.judges.adapters.utils import (
    ChatCompletionError,
    is_response_format_error,
    send_chat_request,
)
from mlflow.genai.judges.tools import list_judge_tools
from mlflow.genai.judges.utils.parsing_utils import (
    _sanitize_justification,
    _strip_markdown_code_blocks,
)
from mlflow.genai.judges.utils.tool_calling_utils import (
    _process_tool_calls,
    _raise_iteration_limit_exceeded,
    _remove_oldest_tool_call_pair,
)
from mlflow.genai.utils.message_utils import pydantic_to_response_format
from mlflow.metrics.genai.model_utils import (
    _call_llm_provider_api,
    _get_provider_instance,
    _parse_model_uri,
    get_endpoint_type,
    score_model_on_payload,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.utils.workspace_context import get_request_workspace
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

_logger = logging.getLogger(__name__)


# Global cache to track model capabilities across function calls
_MODEL_RESPONSE_FORMAT_CAPABILITIES: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InvokeOutput:
    response: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None


# ---------------------------------------------------------------------------
# MLflow Gateway provider (lightweight provider-like object)
# ---------------------------------------------------------------------------


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
    # Duplicated in litellm_adapter.py — cannot be shared via tool_calling_utils.py
    # because judges.base imports judges.utils.__init__ which imports both adapters,
    # creating a circular import chain.
    from mlflow.genai.judges.base import Judge

    output_fields = Judge.get_output_fields()
    field_definitions = {}
    for field in output_fields:
        field_definitions[field.name] = (str, pydantic.Field(description=field.description))
    return pydantic.create_model("JudgeEvaluation", **field_definitions)


def _build_request(
    messages: list["ChatMessage"],
    tools: list[dict[str, Any]] | None,
    response_format: type[pydantic.BaseModel] | None,
    include_response_format: bool,
    inference_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build an OpenAI-format chat completions request payload."""
    payload: dict[str, Any] = {
        "messages": [_message_to_dict(msg) for msg in messages],
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    if include_response_format:
        schema_cls = response_format or _get_default_judge_response_schema()
        # OpenAI's structured output API requires "additionalProperties": false
        # and "strict": true for json_schema response format. LiteLLM adds these
        # automatically when given a Pydantic model; we must add them explicitly.
        schema = schema_cls.model_json_schema()
        schema["additionalProperties"] = False
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    if inference_params:
        payload.update(inference_params)

    return payload


# ---------------------------------------------------------------------------
# Context window management
# ---------------------------------------------------------------------------


def _get_max_context_tokens(provider: str, model: str) -> int | None:
    """Look up the max input token limit for a model from the vendored model catalog."""
    if info := _lookup_model_info(model, custom_llm_provider=provider):
        return info.get("max_input_tokens")
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


def _parse_response_message(
    response_data: dict[str, Any],
    provider: "BaseProvider",
) -> tuple["ChatMessage", dict[str, Any]]:
    """Parse the assistant message and usage from a chat completions response.

    Uses the provider's adapter to transform the raw response into a
    normalized format, then extracts the assistant message and usage metadata.

    Returns:
        Tuple of (ChatMessage, usage_dict) where usage_dict has
        "prompt_tokens" and "completion_tokens" keys.
    """
    # Lazy import: mlflow.types.llm → mlflow.types.schema → numpy, which breaks
    # the skinny client. Must stay inside the function.
    from mlflow.types.llm import ChatMessage

    chat_response = provider.adapter_class.model_to_chat(response_data, provider.config)
    if not chat_response.choices:
        raise MlflowException("Empty choices in chat completions response")

    resp_msg = chat_response.choices[0].message
    content = resp_msg.content
    # Flatten list content (e.g. Anthropic text blocks) to string
    if isinstance(content, list):
        content = "\n".join(part.text for part in content if hasattr(part, "text") and part.text)
        content = content or None

    tool_calls_raw = None
    if resp_msg.tool_calls:
        # Convert gateway ToolCall objects back to dicts for ChatMessage auto-conversion
        tool_calls_raw = [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in resp_msg.tool_calls
        ]

    # ChatMessage requires content when tool_calls is absent
    if content is None and not tool_calls_raw:
        content = ""

    usage = {}
    if chat_response.usage:
        usage = {
            "prompt_tokens": getattr(chat_response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(chat_response.usage, "completion_tokens", None),
        }

    return (
        ChatMessage(
            role=resp_msg.role,
            content=content,
            tool_calls=tool_calls_raw,
        ),
        usage,
    )


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

    Args:
        model_uri: The full model URI.
        provider: The provider name.
        prompt: The prompt to evaluate. Either a string or a list of message dicts.
        inference_params: Optional dictionary of inference parameters to pass to the
            model (e.g., temperature, top_p, max_tokens).
        response_format: Optional Pydantic model class for structured output.
            Only used for ChatMessage-style prompts.
        base_url: Optional base URL to route requests through.
        extra_headers: Optional extra HTTP headers.

    Returns:
        The JSON response string from the model.

    Raises:
        MlflowException: If the provider is not supported or invocation fails.
    """
    if provider == "gateway":
        workspace_headers: dict[str, str] = {}
        if ws := get_request_workspace():
            workspace_headers[WORKSPACE_HEADER_NAME] = ws
        extra_headers = {**workspace_headers, **(extra_headers or {})}

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
    rf_dict = pydantic_to_response_format(response_format) if response_format else None
    return _call_llm_provider_api(
        provider,
        model_name,
        messages=prompt,
        eval_parameters=inference_params,
        extra_headers=extra_headers,
        proxy_url=base_url,
        response_format=rf_dict,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GatewayAdapter(BaseJudgeAdapter):
    """Adapter for native AI Gateway providers."""

    @classmethod
    def is_applicable(
        cls,
        model_uri: str,
        prompt: str | list["ChatMessage"],
    ) -> bool:
        model_provider, _ = _parse_model_uri(model_uri)
        if not is_supported_provider(model_provider) and model_provider not in {
            "endpoints",
            "gateway",
        }:
            return False
        # "endpoints" (Databricks model serving) only supports string prompts
        # via score_model_on_payload; _get_provider_instance doesn't handle it.
        if isinstance(prompt, list) and model_provider == "endpoints":
            return False
        return True

    def _invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
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

        # When a trace is provided, use the tool-calling loop.
        # "endpoints" (deployment endpoints) don't support provider-based tool calling.
        if input_params.trace is not None:
            if input_params.model_provider == "endpoints":
                raise MlflowException(
                    "Trace-based tool calling is not supported for deployment endpoints "
                    "(endpoints:/...). Use a direct provider URI (e.g. openai:/gpt-4) instead.",
                    error_code=BAD_REQUEST,
                )
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
            response_dict = json.loads(cleaned_response, strict=False)
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
        provider, model_name = _parse_model_uri(model_uri)

        output = self._invoke_and_handle_tools(
            provider=provider,
            model_name=model_name,
            messages=messages,
            trace=trace,
            num_retries=num_retries,
            response_format=output_schema,
            inference_params=inference_params,
        )

        cleaned_response = _strip_markdown_code_blocks(output.response)
        try:
            response_dict = json.loads(cleaned_response, strict=False)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse response from judge model. Response: {output.response}",
                error_code=BAD_REQUEST,
            ) from e
        return output_schema(**response_dict)

    def _invoke_with_tools(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        """Invoke the judge model with trace-based tool calling support."""
        # Lazy import: mlflow.types.llm pulls in numpy via mlflow.types.schema
        from mlflow.types.llm import ChatMessage

        messages = (
            [ChatMessage(role="user", content=input_params.prompt)]
            if isinstance(input_params.prompt, str)
            else input_params.prompt
        )

        output = self._invoke_and_handle_tools(
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
            response_dict = json.loads(cleaned_response, strict=False)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse response from judge model. Response: {output.response}",
                error_code=BAD_REQUEST,
            ) from e

        metadata = {}
        if output.num_prompt_tokens is not None:
            metadata[AssessmentMetadataKey.JUDGE_INPUT_TOKENS] = output.num_prompt_tokens
        if output.num_completion_tokens is not None:
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

        return AdapterInvocationOutput(
            feedback=feedback,
            request_id=output.request_id,
            num_prompt_tokens=output.num_prompt_tokens,
            num_completion_tokens=output.num_completion_tokens,
        )

    # TODO: Consider extending _call_llm_provider_api to accept `tools` and return
    # the full ChatResponse (not just text). This would allow replacing the custom
    # send_chat_request + provider adapter calls below with a single shared util.
    def _invoke_and_handle_tools(
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
        """Run the tool-calling loop using the provider infrastructure for HTTP calls."""
        # Lazy import: mlflow.types.llm pulls in numpy via mlflow.types.schema
        from mlflow.types.llm import ChatMessage

        # Resolve provider for config, URL, headers, and request/response transformation.
        # Each provider's get_endpoint_url() returns the full endpoint path
        # (e.g. OpenAI: .../chat/completions, Anthropic: .../messages).
        provider_instance = _get_provider_instance(provider, model_name, base_url=base_url)
        endpoint = base_url or provider_instance.get_endpoint_url("llm/v1/chat")
        headers = dict(provider_instance.headers or {})
        # Tag gateway requests so the server can attribute traffic to the judge
        if provider == "gateway":
            headers[MLFLOW_GATEWAY_CALLER_HEADER] = GatewayCaller.JUDGE.value
            if ws := get_request_workspace():
                headers[WORKSPACE_HEADER_NAME] = ws
        if extra_headers:
            headers.update(extra_headers)

        judge_messages = [ChatMessage(role=msg.role, content=msg.content) for msg in messages]

        tools = []
        if trace is not None:
            judge_tools = list_judge_tools()
            tools = [tool.get_definition().to_dict() for tool in judge_tools]

        include_response_format = _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(
            f"{provider}/{model_name}", True
        )

        max_context_tokens = _get_max_context_tokens(provider, model_name)
        max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
        iteration_count = 0

        while True:
            iteration_count += 1
            if iteration_count > max_iterations:
                _raise_iteration_limit_exceeded(max_iterations)

            try:
                payload = _build_request(
                    messages=judge_messages,
                    tools=tools or None,
                    response_format=response_format,
                    include_response_format=include_response_format,
                    inference_params=inference_params,
                )

                # Use the provider adapter to transform the request
                # (e.g. adds model name, converts tool format for Anthropic)
                chat_payload = provider_instance.adapter_class.chat_to_model(
                    payload, provider_instance.config
                )

                try:
                    response_data = send_chat_request(
                        endpoint=endpoint,
                        headers=headers,
                        payload=chat_payload,
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

                    if (
                        e.status_code == 400
                        and include_response_format
                        and is_response_format_error(e.message)
                    ):
                        _logger.debug(
                            f"Model {provider}/{model_name} may not support structured "
                            f"outputs. Error: {e.message}. Falling back to unstructured "
                            f"response.",
                        )
                        _MODEL_RESPONSE_FORMAT_CAPABILITIES[f"{provider}/{model_name}"] = False
                        include_response_format = False
                        continue

                    # Map HTTP status codes to appropriate MLflow error codes
                    if e.status_code == 400:
                        error_code = BAD_REQUEST
                    elif e.status_code in (401, 403):
                        error_code = INVALID_PARAMETER_VALUE
                    else:
                        error_code = INTERNAL_ERROR

                    raise MlflowException(
                        f"Failed to invoke judge model: {e.message}",
                        error_code=error_code,
                    ) from e

                # Use the provider adapter to normalize the response
                message, usage = _parse_response_message(response_data, provider_instance)

                if not message.tool_calls:
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
