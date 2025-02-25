import logging

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.utils import set_span_chat_messages, set_span_chat_tools
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
    from groq.types.chat.chat_completion import ChatCompletion

    config = AutoLoggingConfig.init(flavor_name=mlflow.groq.FLAVOR_NAME)

    if config.log_traces:
        with mlflow.start_span(
            name=f"{self.__class__.__name__}",
            span_type=_get_span_type(self.__class__),
        ) as span:
            span.set_inputs(kwargs)

            if tools := kwargs.get("tools"):
                try:
                    set_span_chat_tools(span, tools)
                except Exception:
                    _logger.debug(f"Failed to set tools for {span}.", exc_info=True)

            outputs = original(self, *args, **kwargs)
            span.set_outputs(outputs)

            if isinstance(outputs, ChatCompletion):
                try:
                    messages = kwargs.get("messages", [])
                    set_span_chat_messages(
                        span, [*messages, outputs.choices[0].message.model_dump()]
                    )
                except Exception:
                    _logger.debug(f"Failed to set chat messages for {span}.", exc_info=True)

            return outputs
