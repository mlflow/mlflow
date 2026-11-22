import functools
import inspect
import logging
from typing import Any

from pydantic import BaseModel

import mlflow
import mlflow.mistral
from mlflow.entities import SpanType
from mlflow.mistral.chat import convert_tool_to_mlflow_chat_tool
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.distributed import _get_tracing_headers_from_span
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.utils import (
    set_span_chat_tools,
    set_span_model_attribute,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def _construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments


def patched_class_call(original, self, *args, **kwargs):
    """Synchronous wrapper that traces Mistral SDK calls using a context manager."""
    with TracingSession(original, self, args, kwargs) as manager:
        _inject_tracing_headers(kwargs, manager.span)
        output = original(self, *args, **kwargs)
        manager.output = output
        return output


async def async_patched_class_call(original, self, *args, **kwargs):
    """Async wrapper that traces Mistral SDK calls using a context manager."""
    async with TracingSession(original, self, args, kwargs) as manager:
        _inject_tracing_headers(kwargs, manager.span)
        output = await original(self, *args, **kwargs)
        manager.output = output
        return output


def patched_stream_call(original, self, *args, **kwargs):
    """
    Synchronous wrapper for ``Chat.stream``. The span stays open until the returned
    stream is exhausted or closed, and records the completion assembled from its chunks.
    """
    with TracingSession(original, self, args, kwargs) as manager:
        _inject_tracing_headers(kwargs, manager.span)
        stream = original(self, *args, **kwargs)
        manager.output = stream
        manager.wrap_stream(stream)
        return stream


async def async_patched_stream_call(original, self, *args, **kwargs):
    """Async wrapper for ``Chat.stream_async``. See ``patched_stream_call``."""
    async with TracingSession(original, self, args, kwargs) as manager:
        _inject_tracing_headers(kwargs, manager.span)
        stream = await original(self, *args, **kwargs)
        manager.output = stream
        manager.wrap_stream(stream)
        return stream


class TracingSession:
    """Context manager for handling MLflow spans in both sync and async contexts."""

    def __init__(self, original, instance, args, kwargs):
        self.original = original
        self.instance = instance
        self.inputs = _construct_full_inputs(original, instance, *args, **kwargs)

        # These attributes are set outside the constructor.
        self.span = None
        self.token = None
        self.output = None
        self.is_streaming = False

    def __enter__(self):
        return self._enter_impl()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self._enter_impl()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    def _enter_impl(self):
        config = AutoLoggingConfig.init(flavor_name=mlflow.mistral.FLAVOR_NAME)
        if not config.log_traces:
            return self

        self.span = mlflow.start_span_no_context(
            name=f"{self.instance.__class__.__name__}.{self.original.__name__}",
            span_type=SpanType.CHAT_MODEL,
            inputs=self.inputs,
            attributes={SpanAttributeKey.MESSAGE_FORMAT: "mistral"},
        )

        if (tools := self.inputs.get("tools")) is not None:
            try:
                tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools if tool]
                set_span_chat_tools(self.span, tools)
            except Exception as e:
                _logger.debug(f"Failed to set tools for {self.span}. Error: {e}")

        # Attach the span to the current context. A single SDK call can create child spans.
        self.token = set_span_in_context(self.span)
        return self

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        if not self.span:
            return

        # Detach span from the context first to avoid leaking the context on errors.
        detach_span_from_context(self.token)

        if exc_val:
            self.span.record_exception(exc_val)
        elif self.is_streaming:
            # The stream wrapper ends the span once the stream is consumed or closed.
            return

        try:
            usage = _parse_usage(self.output)
        except Exception as e:
            usage = None
            _logger.debug(
                f"Failed to extract token usage for span {self.span.name}: {e}",
                exc_info=True,
            )

        # End the span with captured outputs. Keep original object for backward compatibility.
        self._end_span(self.output, usage)

    def _end_span(self, outputs, usage: dict[str, int] | None) -> None:
        set_span_model_attribute(self.span, self.inputs)
        if usage:
            self.span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
        self.span.end(outputs=outputs)

    def wrap_stream(self, stream) -> None:
        """
        Record the events of a Mistral ``EventStream`` / ``EventStreamAsync`` as they are
        consumed, and end the span when the stream is exhausted, closed, or raises.
        """
        if not self.span:
            return

        chunks = []
        finished = False

        def finish(error: BaseException | None = None):
            nonlocal finished
            if finished:
                return
            finished = True
            if error is not None:
                self.span.record_exception(error)
                self._end_span(None, None)
                return
            try:
                outputs = _reconstruct_completion_from_chunks(chunks)
                usage = next(filter(None, map(_parse_usage, reversed(chunks))), None)
            except Exception:
                _logger.debug("Failed to reconstruct the streaming response", exc_info=True)
                outputs, usage = None, None
            self._end_span(outputs, usage)

        # The SDK closes the HTTP response itself when the events run out or fail, while the
        # generator below is still advancing. Those closes are ignored so that the generator
        # ends the span with the right status. Only a close from user code ends it there.
        iterating = False

        def record_events(generator):
            nonlocal iterating
            try:
                while True:
                    iterating = True
                    try:
                        event = next(generator)
                    except StopIteration:
                        return
                    finally:
                        iterating = False
                    if (chunk := getattr(event, "data", None)) is not None:
                        chunks.append(chunk)
                    yield event
            except Exception as e:
                finish(e)
                raise
            finally:
                finish()

        async def record_events_async(generator):
            nonlocal iterating
            try:
                while True:
                    iterating = True
                    try:
                        event = await anext(generator)
                    except StopAsyncIteration:
                        return
                    finally:
                        iterating = False
                    if (chunk := getattr(event, "data", None)) is not None:
                        chunks.append(chunk)
                    yield event
            except Exception as e:
                finish(e)
                raise
            finally:
                finish()

        def finish_on_close():
            if not iterating:
                finish()

        try:
            generator = stream.generator
            response = stream.response
            if is_async := inspect.isasyncgen(generator):
                original_aclose = response.aclose

                @functools.wraps(original_aclose)
                async def aclose():
                    try:
                        return await original_aclose()
                    finally:
                        finish_on_close()

            else:
                original_close = response.close

                @functools.wraps(original_close)
                def close():
                    try:
                        return original_close()
                    finally:
                        finish_on_close()

        except AttributeError:
            # The stream internals changed. Leave the stream untouched and end the span now.
            _logger.debug("Unable to trace the Mistral stream", exc_info=True)
            return

        # ``EventStream.close()`` (mistralai >= 2.0) and ``EventStream.__exit__`` (mistralai < 2.0)
        # both close the HTTP response, so patching it covers streams that are closed early.
        if is_async:
            stream.generator = record_events_async(generator)
            response.aclose = aclose
        else:
            stream.generator = record_events(generator)
            response.close = close
        self.is_streaming = True


def _inject_tracing_headers(kwargs, span):
    if span is None:
        return
    try:
        if tracing_headers := _get_tracing_headers_from_span(span):
            existing = kwargs.get("http_headers") or {}
            # Replace the mapping so inputs captured before injection keep the user's headers.
            kwargs["http_headers"] = tracing_headers | dict(existing)
    except Exception:
        _logger.debug("Failed to inject tracing headers", exc_info=True)


def _reconstruct_completion_from_chunks(chunks: list[Any]) -> dict[str, Any] | None:
    """
    Assemble a chat completion, in the shape of Mistral's ``ChatCompletionResponse``, from the
    ``CompletionChunk`` objects of a stream.
    """
    if not chunks:
        return None

    content_parts = []
    tool_calls = []
    role = "assistant"
    finish_reason = None
    for chunk in chunks:
        for choice in chunk.choices or []:
            if choice.index != 0:
                continue
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if isinstance(delta.role, str):
                role = delta.role
            _merge_content(content_parts, delta.content)
            for tool_call in delta.tool_calls or []:
                _merge_tool_call(tool_calls, tool_call)

    if all(part["type"] == "text" for part in content_parts):
        content = "".join(part["text"] for part in content_parts)
    else:
        # Reasoning models stream thinking and text chunks, so keep them as a chunk list.
        content = content_parts

    message = {"role": role, "content": content}
    if tool_calls:
        message["tool_calls"] = [tool_call for _, tool_call in tool_calls]

    last_chunk = chunks[-1]
    return {
        "id": last_chunk.id,
        "object": "chat.completion",
        "model": last_chunk.model,
        "created": last_chunk.created,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def _merge_content(parts: list[dict[str, Any]], content: Any) -> None:
    # ``content`` is either a string or a list of content chunks (e.g. text and thinking).
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content = [c.model_dump() if isinstance(c, BaseModel) else c for c in content]
    else:
        return

    for part in content:
        match part, parts[-1] if parts else None:
            case {"type": "text", "text": str(text)}, {"type": "text"} as last:
                last["text"] += text
            case {"type": "thinking", "thinking": list(thinking)}, {
                "type": "thinking",
                "thinking": [{"type": "text"} as last, *_],
            }:
                last["text"] += "".join(t.get("text") or "" for t in thinking)
            case {"type": "thinking", "thinking": list(thinking)}, _:
                text = "".join(t.get("text") or "" for t in thinking)
                parts.append({"type": "thinking", "thinking": [{"type": "text", "text": text}]})
            case _:
                parts.append(dict(part))


def _merge_tool_call(tool_calls: list[tuple[int, dict[str, Any]]], tool_call: Any) -> None:
    # Mistral usually sends each tool call whole in one chunk. A later chunk that has the same
    # index and no new ID continues the previous call, so its arguments are appended.
    index = getattr(tool_call, "index", None) or 0
    call_id = tool_call.id if tool_call.id not in (None, "null") else None
    match tool_calls[-1] if tool_calls else None:
        case (last_index, last) if last_index == index and call_id in (None, last.get("id")):
            arguments = tool_call.function.arguments
            if isinstance(arguments, str) and arguments:
                last["function"]["arguments"] = (
                    last["function"].get("arguments") or ""
                ) + arguments
        case _:
            tool_calls.append((index, tool_call.model_dump(exclude={"index"})))


def _parse_usage(output):
    usage = getattr(output, "usage", None)
    if usage is None:
        return None

    usage_dict = {}
    if getattr(usage, "prompt_tokens", None) is not None:
        usage_dict[TokenUsageKey.INPUT_TOKENS] = usage.prompt_tokens
    if getattr(usage, "completion_tokens", None) is not None:
        usage_dict[TokenUsageKey.OUTPUT_TOKENS] = usage.completion_tokens
    if getattr(usage, "total_tokens", None) is not None:
        usage_dict[TokenUsageKey.TOTAL_TOKENS] = usage.total_tokens

    return usage_dict or None
