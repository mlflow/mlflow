import io
import json
import logging
from typing import Any, Optional, Union

from botocore.client import BaseClient
from botocore.response import StreamingBody

import mlflow
from mlflow.bedrock import FLAVOR_NAME
from mlflow.bedrock.chat import convert_message_to_mlflow_chat, convert_tool_to_mlflow_chat_tool
from mlflow.bedrock.stream import ConverseStreamWrapper, InvokeModelStreamWrapper
from mlflow.bedrock.utils import skip_if_trace_disabled
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.utils import set_span_chat_messages, set_span_chat_tools
from mlflow.utils.autologging_utils import safe_patch

_BEDROCK_RUNTIME_SERVICE_NAME = "bedrock-runtime"
_BEDROCK_SPAN_PREFIX = "BedrockRuntime."

_logger = logging.getLogger(__name__)


def patched_create_client(original, self, *args, **kwargs):
    """
    Patched version of the boto3 ClientCreator.create_client method that returns
    a patched client class.
    """
    if kwargs.get("service_name") != _BEDROCK_RUNTIME_SERVICE_NAME:
        return original(self, *args, **kwargs)

    client = original(self, *args, **kwargs)
    patch_bedrock_runtime_client(client.__class__)

    return client


def patch_bedrock_runtime_client(client_class: type[BaseClient]):
    """
    Patch the BedrockRuntime client to log traces and models.
    """
    # The most basic model invocation API
    safe_patch(FLAVOR_NAME, client_class, "invoke_model", _patched_invoke_model)
    safe_patch(
        FLAVOR_NAME,
        client_class,
        "invoke_model_with_response_stream",
        _patched_invoke_model_with_response_stream,
    )

    if hasattr(client_class, "converse"):
        # The new "converse" API was introduced in boto3 1.35 to access all models
        # with the consistent chat format.
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
        safe_patch(FLAVOR_NAME, client_class, "converse", _patched_converse)

    if hasattr(client_class, "converse_stream"):
        safe_patch(FLAVOR_NAME, client_class, "converse_stream", _patched_converse_stream)


@skip_if_trace_disabled
def _patched_invoke_model(original, self, *args, **kwargs):
    with mlflow.start_span(name=f"{_BEDROCK_SPAN_PREFIX}{original.__name__}") as span:
        # NB: Bedrock client doesn't accept any positional arguments
        span.set_inputs(kwargs)

        result = original(self, *args, **kwargs)

        result["body"] = _buffer_stream(result["body"])
        parsed_response_body = _parse_invoke_model_response_body(result["body"])

        # Determine the span type based on the key in the response body.
        # As of 2024 Dec 9th, all supported embedding models in Bedrock returns the response body
        # with the key "embedding". This might change in the future.
        span_type = SpanType.EMBEDDING if "embedding" in parsed_response_body else SpanType.LLM
        span.set_span_type(span_type)

        # Record token usage if provided in response
        usage = _parse_usage_from_response(parsed_response_body)
        if usage:
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)

        span.set_outputs({**result, "body": parsed_response_body})

        return result


@skip_if_trace_disabled
def _patched_invoke_model_with_response_stream(original, self, *args, **kwargs):
    span = start_span_no_context(
        name=f"{_BEDROCK_SPAN_PREFIX}{original.__name__}",
        # NB: Since we don't inspect the response body for this method, the span type is unknown.
        # We assume it is LLM as using streaming for embedding is not common.
        span_type=SpanType.LLM,
        inputs=kwargs,
    )

    result = original(self, *args, **kwargs)

    # To avoid consuming the stream during serialization, set dummy outputs for the span.
    span.set_outputs({**result, "body": "EventStream"})

    # Wrap streaming body to capture usage events
    result_body = InvokeModelStreamWrapper(stream=result["body"], span=span)
    # After stream ends, _buffer_usage events will set CHAT_USAGE attribute
    result["body"] = result_body
    return result


def _buffer_stream(raw_stream: StreamingBody) -> StreamingBody:
    """
    Create a buffered stream from the raw byte stream.

    The boto3's invoke_model() API returns the LLM response as a byte stream.
    We need to read the stream data to set the span outputs, however, the stream
    can only be read once and not seekable (https://github.com/boto/boto3/issues/564).
    To work around this, we create a buffered stream that can be read multiple times.
    """
    buffered_response = io.BytesIO(raw_stream.read())
    buffered_response.seek(0)
    return StreamingBody(buffered_response, raw_stream._content_length)


def _parse_invoke_model_response_body(response_body: StreamingBody) -> Union[dict[str, Any], str]:
    content = response_body.read()
    try:
        return json.loads(content)
    except Exception:
        # When failed to parse the response body as JSON, return the raw response
        return content
    finally:
        # Reset the stream position to the beginning
        response_body._raw_stream.seek(0)
        # Boto3 uses this attribute to validate the amount of data read from the stream matches
        # the content length, so we need to reset it as well.
        # https://github.com/boto/botocore/blob/f88e981cb1a6cd0c64bc89da262ab76f9bfa9b7d/botocore/response.py#L164C17-L164C32
        response_body._amount_read = 0


def _parse_usage_from_response(
    response_body: Union[dict[str, Any], str],
) -> Optional[dict[str, int]]:
    """
    Parse token usage information from Bedrock response body.

    Different Bedrock models return usage information in different formats:
    - Anthropic: {"usage": {"input_tokens": int, "output_tokens": int}}
    - AI21: {"usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}}
    - Amazon Nova: {"usage": {"inputTokens": int, "outputTokens": int, "totalTokens": int}}
    - Meta Llama: {"prompt_token_count": int, "generation_token_count": int}
    """
    if not isinstance(response_body, dict):
        return None

    try:
        # Handle Anthropic format
        if "usage" in response_body and isinstance(response_body["usage"], dict):
            usage = response_body["usage"]
            if "input_tokens" in usage and "output_tokens" in usage:
                return {
                    TokenUsageKey.INPUT_TOKENS: usage["input_tokens"],
                    TokenUsageKey.OUTPUT_TOKENS: usage["output_tokens"],
                    TokenUsageKey.TOTAL_TOKENS: usage.get(
                        "total_tokens",
                        usage["input_tokens"] + usage["output_tokens"],
                    ),
                }
            # Handle AI21 format
            elif "prompt_tokens" in usage and "completion_tokens" in usage:
                return {
                    TokenUsageKey.INPUT_TOKENS: usage["prompt_tokens"],
                    TokenUsageKey.OUTPUT_TOKENS: usage["completion_tokens"],
                    TokenUsageKey.TOTAL_TOKENS: usage.get(
                        "total_tokens",
                        usage["prompt_tokens"] + usage["completion_tokens"],
                    ),
                }
            # Handle Amazon Nova format
            elif "inputTokens" in usage and "outputTokens" in usage:
                return {
                    TokenUsageKey.INPUT_TOKENS: usage["inputTokens"],
                    TokenUsageKey.OUTPUT_TOKENS: usage["outputTokens"],
                    TokenUsageKey.TOTAL_TOKENS: usage.get(
                        "totalTokens",
                        usage["inputTokens"] + usage["outputTokens"],
                    ),
                }
        # Handle Meta Llama format (top-level fields)
        elif "prompt_token_count" in response_body and "generation_token_count" in response_body:
            return {
                TokenUsageKey.INPUT_TOKENS: response_body["prompt_token_count"],
                TokenUsageKey.OUTPUT_TOKENS: response_body["generation_token_count"],
                TokenUsageKey.TOTAL_TOKENS: (
                    response_body["prompt_token_count"] + response_body["generation_token_count"]
                ),
            }
        # Add debug log for unknown usage schema
        _logger.debug(f"Unknown token usage schema in Bedrock response: {response_body}")
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from response: {e}")

    return None


@skip_if_trace_disabled
def _patched_converse(original, self, *args, **kwargs):
    with mlflow.start_span(
        name=f"{_BEDROCK_SPAN_PREFIX}{original.__name__}",
        span_type=SpanType.CHAT_MODEL,
    ) as span:
        # NB: Bedrock client doesn't accept any positional arguments
        span.set_inputs(kwargs)
        _set_tool_attributes(span, kwargs)

        result = None
        try:
            result = original(self, *args, **kwargs)
            span.set_outputs(result)
            # Use _parse_usage_from_response for all usage extraction
            usage = _parse_usage_from_response(result)
            if usage:
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
        finally:
            _set_chat_messages_attributes(span, kwargs.get("messages", []), result)
        return result


@skip_if_trace_disabled
def _patched_converse_stream(original, self, *args, **kwargs):
    # NB: Do not use fluent API to create a span for streaming response. If we do so,
    # the span context will remain active until the stream is fully exhausted, which
    # can lead to super hard-to-debug issues.
    span = start_span_no_context(
        name=f"{_BEDROCK_SPAN_PREFIX}{original.__name__}",
        span_type=SpanType.CHAT_MODEL,
        inputs=kwargs,
    )
    _set_tool_attributes(span, kwargs)

    result = original(self, *args, **kwargs)

    if span:
        result["stream"] = ConverseStreamWrapper(
            stream=result["stream"],
            span=span,
            inputs=kwargs,
        )

    return result


def _set_chat_messages_attributes(span, messages: list[dict], response: Optional[dict]):
    """
    Extract standard chat span attributes for the Bedrock Converse API call.

    NB: We only support standard attribute extraction for the Converse API, because
    the InvokeModel API exposes the raw API spec from each LLM provider, hence
    maintaining the compatibility for all providers is significantly cumbersome.
    """
    try:
        messages = [*messages]  # shallow copy to avoid appending to the original list
        if response:
            messages.append(response["output"]["message"])
        messages = [convert_message_to_mlflow_chat(msg) for msg in messages]
        set_span_chat_messages(span, messages)
    except Exception as e:
        _logger.debug(f"Failed to set messages for {span}. Error: {e}")


def _set_tool_attributes(span, kwargs):
    """Extract tool attributes for the Bedrock Converse API call."""
    if tool_config := kwargs.get("toolConfig"):
        try:
            tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tool_config["tools"]]
            set_span_chat_tools(span, tools)
        except Exception as e:
            _logger.debug(f"Failed to set tools for {span}. Error: {e}")
