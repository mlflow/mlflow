import logging
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.distributed import _get_tracing_headers_from_span
from mlflow.tracing.utils import set_span_chat_tools
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def _get_span_type(resource: type) -> str:
    from groq.resources.audio.transcriptions import Transcriptions
    from groq.resources.audio.translations import Translations
    from groq.resources.chat.completions import Completions
    from groq.resources.embeddings import Embeddings

    span_type_mapping = {
        Completions: SpanType.CHAT_MODEL,
        Transcriptions: SpanType.LLM,
        Translations: SpanType.LLM,
        Embeddings: SpanType.EMBEDDING,
    }
    return span_type_mapping.get(resource, SpanType.UNKNOWN)


def patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.groq.FLAVOR_NAME)

    if config.log_traces:
        with mlflow.start_span(
            name=f"{self.__class__.__name__}",
            span_type=_get_span_type(self.__class__),
        ) as span:
            span.set_inputs(kwargs)
            span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "groq")

            # Extract model name from kwargs
            if model := kwargs.get("model"):
                span.set_attribute(SpanAttributeKey.MODEL, model)
                span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, "groq")

            if tools := kwargs.get("tools"):
                try:
                    set_span_chat_tools(span, tools)
                except Exception:
                    _logger.debug(f"Failed to set tools for {span}.", exc_info=True)

            _inject_tracing_headers(kwargs, span)
            outputs = original(self, *args, **kwargs)
            span.set_outputs(outputs)

            if usage := _parse_usage(outputs):
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)

            return outputs


def _inject_tracing_headers(kwargs, span):
    try:
        if tracing_headers := _get_tracing_headers_from_span(span):
            existing = kwargs.get("extra_headers") or {}
            kwargs["extra_headers"] = tracing_headers | dict(existing)
    except Exception:
        _logger.debug("Failed to inject tracing headers", exc_info=True)


def _parse_usage(output: Any) -> dict[str, int] | None:
    if (usage := getattr(output, "usage", None)) is None:
        return None

    # Embedding usage has no `completion_tokens`, so read each field only when present.
    usage_dict = {}
    if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
        usage_dict[TokenUsageKey.INPUT_TOKENS] = prompt_tokens
    if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
        usage_dict[TokenUsageKey.OUTPUT_TOKENS] = completion_tokens
    if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
        usage_dict[TokenUsageKey.TOTAL_TOKENS] = total_tokens
    return usage_dict or None
