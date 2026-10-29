import functools
import logging
from typing import Any, Iterator

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.distributed import _get_tracing_headers_from_span
from mlflow.tracing.fluent import start_span_no_context
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
    from groq import Stream

    config = AutoLoggingConfig.init(flavor_name=mlflow.groq.FLAVOR_NAME)

    if config.log_traces:
        # Use a detached span so that a streaming response can end it after this call returns.
        span = start_span_no_context(
            name=f"{self.__class__.__name__}",
            span_type=_get_span_type(self.__class__),
            inputs=kwargs,
            attributes={SpanAttributeKey.MESSAGE_FORMAT: "groq"},
        )

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
        try:
            outputs = original(self, *args, **kwargs)
        except Exception as e:
            _end_span_on_exception(span, e)
            raise

        if isinstance(outputs, Stream):
            _wrap_stream(outputs, span)
        else:
            _end_span(span, outputs, _parse_usage(outputs))

        return outputs


def _end_span(span: LiveSpan, outputs: Any, usage: dict[str, int] | None):
    if usage:
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
    span.end(outputs=outputs)


def _end_span_on_exception(span: LiveSpan, e: BaseException):
    try:
        span.add_event(SpanEvent.from_exception(e))
        span.end(status=SpanStatusCode.ERROR)
    except Exception:
        _logger.debug("Failed to end span on exception", exc_info=True)


def _wrap_stream(stream, span: LiveSpan):
    """
    Record the chunks of a streaming response and end the span when the stream is
    exhausted or closed, instead of when ``create`` returns.
    """
    chunks = []
    finished = False

    def _finish(error=None):
        nonlocal finished
        if finished:
            return
        finished = True
        if error is not None:
            _end_span_on_exception(span, error)
            return
        try:
            outputs = _reconstruct_completion_from_chunks(chunks)
            usage = next(filter(None, map(_parse_chunk_usage, reversed(chunks))), None)
        except Exception:
            _logger.debug("Failed to reconstruct the streaming response", exc_info=True)
            outputs, usage = None, None
        _end_span(span, outputs, usage)

    def _record_chunks(iterator: Iterator) -> Iterator:
        try:
            for chunk in iterator:
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            _finish(e)
            raise
        finally:
            _finish()

    original_close = stream.close

    @functools.wraps(original_close)
    def _close():
        try:
            return original_close()
        finally:
            _finish()

    stream._iterator = _record_chunks(stream._iterator)
    stream.close = _close


def _reconstruct_completion_from_chunks(chunks: list[Any]) -> dict[str, Any] | None:
    if not chunks:
        return None

    content = ""
    reasoning = ""
    tool_calls = {}
    finish_reason = None
    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason or finish_reason
        content += choice.delta.content or ""
        reasoning += getattr(choice.delta, "reasoning", None) or ""
        for tool_call in choice.delta.tool_calls or []:
            if tool_call.index not in tool_calls:
                tool_calls[tool_call.index] = tool_call.model_dump(exclude={"index"})
                continue
            merged = tool_calls[tool_call.index]["function"]
            if merged and tool_call.function and tool_call.function.arguments:
                merged["arguments"] = (merged["arguments"] or "") + tool_call.function.arguments

    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

    last_chunk = chunks[-1]
    return {
        "id": last_chunk.id,
        "object": "chat.completion",
        "created": last_chunk.created,
        "model": last_chunk.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def _inject_tracing_headers(kwargs, span):
    try:
        if tracing_headers := _get_tracing_headers_from_span(span):
            existing = kwargs.get("extra_headers") or {}
            kwargs["extra_headers"] = tracing_headers | dict(existing)
    except Exception:
        _logger.debug("Failed to inject tracing headers", exc_info=True)


def _parse_chunk_usage(chunk: Any) -> dict[str, int] | None:
    # Groq reports the usage of a streaming response in ``x_groq.usage`` of the last chunk,
    # or in ``usage`` when ``stream_options={"include_usage": True}`` is set (via ``extra_body``).
    return _parse_usage(chunk) or _parse_usage(getattr(chunk, "x_groq", None))


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
