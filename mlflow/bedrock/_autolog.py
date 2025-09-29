import io
import json
import logging
from typing import Any

from botocore.client import BaseClient
from botocore.response import StreamingBody

import mlflow
from mlflow.bedrock import FLAVOR_NAME
from mlflow.bedrock.chat import convert_tool_to_mlflow_chat_tool
from mlflow.bedrock.stream import ConverseStreamWrapper, InvokeModelStreamWrapper
from mlflow.bedrock.utils import parse_complete_token_usage_from_response, skip_if_trace_disabled
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.utils import set_span_chat_tools
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


def _parse_usage_from_response(
    response_data: dict[str, Any] | str,
) -> dict[str, int] | None:
    """Parse token usage from Bedrock API response body.

    Args:
        response_data: The response body from Bedrock API, either as dict or string.

    Returns:
        Standardized token usage dictionary, or None if parsing fails or no usage found.
    """
    try:
        if isinstance(response_data, dict):
            if usage_data := response_data.get("usage"):
                return parse_complete_token_usage_from_response(usage_data)

            # If no "usage" field, check if the response itself contains token fields
            # (e.g., Meta Llama responses have prompt_token_count, generation_token_count)
            return parse_complete_token_usage_from_response(response_data)
        return None
    except (KeyError, TypeError, ValueError) as e:
        _logger.debug(f"Failed to parse token usage from response: {e}")
        return None


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
        span.set_outputs({**result, "body": parsed_response_body})

        # Parse and set token usage information if available
        if usage_data := _parse_usage_from_response(parsed_response_body):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_data)

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

    result["body"] = InvokeModelStreamWrapper(stream=result["body"], span=span)
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


def _parse_invoke_model_response_body(response_body: StreamingBody) -> dict[str, Any] | str:
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


@skip_if_trace_disabled
def _patched_converse(original, self, *args, **kwargs):
    with mlflow.start_span(
        name=f"{_BEDROCK_SPAN_PREFIX}{original.__name__}",
        span_type=SpanType.CHAT_MODEL,
    ) as span:
        # NB: Bedrock client doesn't accept any positional arguments
        span.set_inputs(kwargs)
        span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "bedrock")
        _set_tool_attributes(span, kwargs)

        result = original(self, *args, **kwargs)
        span.set_outputs(result)

        # Parse and set token usage information if available
        if usage_data := _parse_usage_from_response(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_data)

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
        attributes={SpanAttributeKey.MESSAGE_FORMAT: "bedrock"},
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


def _set_tool_attributes(span, kwargs):
    """Extract tool attributes for the Bedrock Converse API call."""
    if tool_config := kwargs.get("toolConfig"):
        try:
            tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tool_config["tools"]]
            set_span_chat_tools(span, tools)
        except Exception as e:
            _logger.debug(f"Failed to set tools for {span}. Error: {e}")
