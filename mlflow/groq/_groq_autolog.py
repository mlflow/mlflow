import logging
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
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

            if tools := kwargs.get("tools"):
                try:
                    set_span_chat_tools(span, tools)
                except Exception:
                    _logger.debug(f"Failed to set tools for {span}.", exc_info=True)

            outputs = original(self, *args, **kwargs)
            span.set_outputs(outputs)

            if usage := _parse_usage(outputs):
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)

            return outputs


def _parse_usage(output: Any) -> dict[str, int] | None:
    try:
        if usage := getattr(output, "usage", None):
            return {
                TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
            }
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from output: {e}")
    return None
