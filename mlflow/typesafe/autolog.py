import asyncio
import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import mlflow
import mlflow.typesafe
from mlflow.entities import SpanLogLevel, SpanStatus, SpanStatusCode, SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)

_SPAN_NAME = "typesafe.system_one"
_REQUEST_ID_ATTRIBUTE = "typesafe.request_id"


def patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.typesafe.FLAVOR_NAME)
    if not config.log_traces:
        return original(self, *args, **kwargs)

    inputs = _construct_effective_inputs(original, self, args, kwargs)
    with mlflow.start_span(name=_SPAN_NAME, span_type=SpanType.LLM) as span:
        span.set_inputs(inputs)
        _set_request_attributes(span, inputs)
        try:
            output = original(self, *args, **kwargs)
        except Exception as e:
            _set_request_id_attribute(span, e)
            raise
        _set_response_attributes(span, output, inputs)
        return output


async def async_patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.typesafe.FLAVOR_NAME)
    if not config.log_traces:
        return await original(self, *args, **kwargs)

    inputs = _construct_effective_inputs(original, self, args, kwargs)
    with mlflow.start_span(name=_SPAN_NAME, span_type=SpanType.LLM) as span:
        span.set_inputs(inputs)
        _set_request_attributes(span, inputs)
        try:
            output = await original(self, *args, **kwargs)
        except asyncio.CancelledError as e:
            span.set_log_level(SpanLogLevel.ERROR)
            span.set_status(
                SpanStatus(
                    status_code=SpanStatusCode.ERROR,
                    description=f"{type(e).__name__}: {e}",
                )
            )
            raise
        except Exception as e:
            _set_request_id_attribute(span, e)
            raise
        _set_response_attributes(span, output, inputs)
        return output


def _construct_effective_inputs(original, instance, args, kwargs) -> dict[str, Any]:
    """Build the JSON request body while excluding transport and authentication options."""
    try:
        arguments = inspect.signature(original).bind_partial(instance, *args, **kwargs).arguments
    except (TypeError, ValueError):
        arguments = dict(kwargs)
        if args:
            arguments.setdefault("state", args[0])
        if len(args) > 1:
            arguments.setdefault("questions", args[1])

    arguments.pop("self", None)
    inputs = {}
    if "state" in arguments:
        inputs["state"] = arguments["state"]

    requested_model = arguments.get("model")
    if requested_model is None:
        requested_model = getattr(getattr(instance, "_config", None), "default_model", None)
    inputs["model"] = requested_model

    if "questions" in arguments:
        inputs["questions"] = arguments["questions"]

    extra_body = arguments.get("extra_body")
    if isinstance(extra_body, Mapping):
        inputs.update(extra_body)

    return _serialize_for_trace(inputs)


def _serialize_for_trace(value: Any) -> Any:
    if callable(model_dump := getattr(value, "model_dump", None)):
        try:
            return model_dump(mode="json")
        except Exception:
            _logger.debug("Failed to serialize a TypeSafe value with model_dump", exc_info=True)

    if isinstance(value, Mapping):
        return {key: _serialize_for_trace(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_for_trace(item) for item in value]
    return value


def _set_request_attributes(span: LiveSpan, inputs: dict[str, Any]) -> None:
    span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, "typesafe")
    if model := inputs.get("model"):
        span.set_attribute(SpanAttributeKey.MODEL, model)


def _set_response_attributes(span: LiveSpan, output: Any, inputs: dict[str, Any]) -> None:
    serialized_output = _serialize_for_trace(output)
    span.set_outputs(serialized_output)

    try:
        if model := _get_value(output, serialized_output, "model") or inputs.get("model"):
            span.set_attribute(SpanAttributeKey.MODEL, model)
    except Exception:
        _logger.debug("Failed to extract the model from a TypeSafe response", exc_info=True)

    try:
        if usage := _parse_usage(output, serialized_output):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
    except Exception:
        _logger.debug("Failed to extract token usage from a TypeSafe response", exc_info=True)

    _set_request_id_attribute(span, output)


def _get_value(value: Any, serialized_value: Any, key: str) -> Any:
    if isinstance(serialized_value, Mapping):
        return serialized_value.get(key)
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _parse_usage(output: Any, serialized_output: Any) -> dict[str, int] | None:
    usage = _get_value(output, serialized_output, "usage")
    if usage is None:
        return None

    input_tokens = _get_usage_value(usage, TokenUsageKey.INPUT_TOKENS)
    output_tokens = _get_usage_value(usage, TokenUsageKey.OUTPUT_TOKENS)
    total_tokens = _get_usage_value(usage, TokenUsageKey.TOTAL_TOKENS)

    parsed = {}
    if input_tokens is not None:
        parsed[TokenUsageKey.INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        parsed[TokenUsageKey.OUTPUT_TOKENS] = output_tokens
    if total_tokens is not None:
        parsed[TokenUsageKey.TOTAL_TOKENS] = total_tokens
    elif input_tokens is not None and output_tokens is not None:
        parsed[TokenUsageKey.TOTAL_TOKENS] = input_tokens + output_tokens
    return parsed or None


def _get_usage_value(usage: Any, key: str) -> Any:
    if isinstance(usage, Mapping):
        return usage.get(key)
    return getattr(usage, key, None)


def _set_request_id_attribute(span: LiveSpan, value: Any) -> None:
    try:
        request_id = getattr(value, "request_id", None)
    except Exception:
        return
    if request_id:
        span.set_attribute(_REQUEST_ID_ATTRIBUTE, request_id)
