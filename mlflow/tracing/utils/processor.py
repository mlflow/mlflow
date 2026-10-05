import logging

from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import calculate_cost_by_model_and_token_usage, dump_span_attribute_value

_logger = logging.getLogger(__name__)


def preserve_evaluation_span_metrics(otel_span, live_span):
    if live_span.get_attribute(SpanAttributeKey.EVALUATION_SCORER) is not True:
        return

    from mlflow.tracing.otel.translation import (
        _get_model_name,
        _get_model_provider,
        _get_token_usage,
    )

    attributes = dict(otel_span.attributes or {})
    for original, evaluation in (
        (SpanAttributeKey.CHAT_USAGE, SpanAttributeKey.EVALUATION_TOKEN_USAGE),
        (SpanAttributeKey.LLM_COST, SpanAttributeKey.EVALUATION_COST),
    ):
        if original in attributes:
            attributes[evaluation] = attributes.pop(original)

    if SpanAttributeKey.EVALUATION_TOKEN_USAGE not in attributes:
        if usage := _get_token_usage(attributes):
            attributes[SpanAttributeKey.EVALUATION_TOKEN_USAGE] = dump_span_attribute_value(usage)

    if (
        SpanAttributeKey.EVALUATION_COST not in attributes
        and SpanAttributeKey.EVALUATION_TOKEN_USAGE in attributes
    ):
        usage = live_span.get_attribute(SpanAttributeKey.CHAT_USAGE) or _get_token_usage(attributes)
        model = live_span.get_attribute(SpanAttributeKey.MODEL) or _get_model_name(attributes)
        provider = live_span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) or _get_model_provider(
            attributes
        )
        if cost := calculate_cost_by_model_and_token_usage(model, usage, provider):
            attributes[SpanAttributeKey.EVALUATION_COST] = dump_span_attribute_value(cost)

    for key in list(attributes):
        if key.startswith(("gen_ai.usage.", "llm.token_count.", "llm.usage.")):
            attributes[f"mlflow.evaluation.original.{key}"] = attributes.pop(key)

    otel_span._attributes = attributes
    live_span._span._attributes = attributes


def apply_span_processors(span):
    """Apply configured span processors sequentially to the span."""
    from mlflow.tracing.config import get_config

    config = get_config()
    if not config.span_processors:
        return

    non_null_return_processors = []
    for processor in config.span_processors:
        try:
            result = processor(span)
            if result is not None:
                non_null_return_processors.append(processor.__name__)
        except Exception as e:
            _logger.warning(
                f"Span processor {processor.__name__} failed: {e}",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )

    if non_null_return_processors:
        _logger.warning(
            f"Span processors {non_null_return_processors} returned a non-null value, "
            "but it will be ignored. Span processors should not return a value."
        )


def validate_span_processors(span_processors):
    """Validate that the span processor is a valid function."""
    span_processors = span_processors or []

    for span_processor in span_processors:
        if not callable(span_processor):
            raise MlflowException.invalid_parameter_value(
                "Span processor must be a callable function."
            )

        # Skip validation for builtin functions and partial functions that don't have __code__
        if not hasattr(span_processor, "__code__"):
            continue

        if span_processor.__code__.co_argcount != 1:
            raise MlflowException.invalid_parameter_value(
                "Span processor must take exactly one argument that accepts a LiveSpan object."
            )

    return span_processors
