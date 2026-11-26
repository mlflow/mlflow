import functools
import inspect
import json
import logging
from typing import Any

from pydantic import BaseModel

import mlflow
import mlflow.mistral
from mlflow.entities import SpanEvent, SpanStatus, SpanStatusCode, SpanType
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
        manager.trace_stream(stream)
        return stream


async def async_patched_stream_call(original, self, *args, **kwargs):
    """Async wrapper for ``Chat.stream_async``. See ``patched_stream_call``."""
    async with TracingSession(original, self, args, kwargs) as manager:
        _inject_tracing_headers(kwargs, manager.span)
        stream = await original(self, *args, **kwargs)
        manager.trace_stream(stream)
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

        if exc_val is None and self.is_streaming:
            # The stream wrapper ends the span once the stream is consumed or closed.
            return

        # Never raise from here: an error in __exit__ / __aexit__ would replace the call's own
        # result or exception, e.g. turn a cancelled call into one that returns None.
        try:
            if exc_val is not None:
                _record_error(self.span, exc_val)
            usage = _parse_usage(self.output)
        except Exception:
            usage = None
            _logger.debug(f"Failed to record the result of span {self.span.name}", exc_info=True)

        # End the span with captured outputs. Keep original object for backward compatibility.
        _end_span(self.span, self.inputs, self.output, usage)

    def trace_stream(self, stream) -> None:
        if self.span is None:
            return
        # The stream is not kept on the session: the wrapper must not reference it, so that
        # dropping the last reference to an unfinished stream finalizes the wrapper right away.
        if _wrap_stream(stream, self.span, self.inputs):
            self.is_streaming = True
        else:
            _warn_untraceable_stream()


def _end_span(span, inputs, outputs, usage: dict[str, int] | None) -> None:
    try:
        set_span_model_attribute(span, inputs)
        if usage:
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
    except Exception:
        _logger.debug(f"Failed to set attributes of span {span.name}", exc_info=True)
    try:
        span.end(outputs=outputs)
    except Exception:
        _logger.debug(f"Failed to end span {span.name}", exc_info=True)


def _record_error(span, error: BaseException) -> None:
    if isinstance(error, Exception):
        span.record_exception(error)
        return
    # ``record_exception`` only accepts ``Exception``, not ``asyncio.CancelledError`` or
    # ``KeyboardInterrupt``, so record those the same way it does.
    span.add_event(SpanEvent.from_exception(error))
    span.set_status(
        SpanStatus(
            status_code=SpanStatusCode.ERROR,
            description=f"{type(error).__name__}: {error}",
        )
    )


@functools.cache
def _warn_untraceable_stream() -> None:
    _logger.warning(
        "Unable to trace a Mistral stream: the installed mistralai version has an unexpected "
        "stream structure. The stream works as usual, but its span has no output or token usage."
    )


def _wrap_stream(stream, span, inputs) -> bool:
    """
    Record the events of a Mistral ``EventStream`` / ``EventStreamAsync`` as they are consumed,
    and end the span when the stream is exhausted, closed, or raises. Returns ``False`` and leaves
    the stream untouched if its internals are not as expected.

    The closures below hold only the span, the inputs and the received chunks, never the stream.
    """
    chunks = []
    finished = False

    def finish(error: BaseException | None = None):
        nonlocal finished
        if finished:
            return
        finished = True
        # On an error, still record what was received before it: it is what you want to see
        # when debugging the failure.
        try:
            outputs = _reconstruct_completion_from_chunks(chunks)
        except Exception:
            outputs = None
            _logger.debug("Failed to reconstruct the streaming response", exc_info=True)
        try:
            usage = next(filter(None, map(_parse_usage, reversed(chunks))), None)
        except Exception:
            usage = None
            _logger.debug("Failed to extract token usage from the stream", exc_info=True)
        if error is not None:
            try:
                _record_error(span, error)
            except Exception:
                _logger.debug("Failed to record the stream error", exc_info=True)
        _end_span(span, inputs, outputs, usage)

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
        except GeneratorExit:
            # Closed early by user code, or garbage collected: the span ends with what was received.
            raise
        except BaseException as e:
            # Errors, but also cancellation and Ctrl-C while waiting for the next event.
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
        except GeneratorExit:
            raise
        except BaseException as e:
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
        _logger.debug("Unable to trace the Mistral stream", exc_info=True)
        return False

    # ``EventStream.close()`` (mistralai >= 2.0) and ``EventStream.__exit__`` (mistralai < 2.0)
    # both close the HTTP response, so patching it covers streams that are closed early.
    if is_async:
        stream.generator = record_events_async(generator)
        response.aclose = aclose
    else:
        stream.generator = record_events(generator)
        response.close = close
    return True


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
    ``CompletionChunk`` objects of a stream. Each choice index (``n > 1``) becomes its own choice.
    """
    if not chunks:
        return None

    choices: dict[int, _StreamedChoice] = {}
    for chunk in chunks:
        for choice in chunk.choices or []:
            choices.setdefault(choice.index, _StreamedChoice()).add(choice)

    last_chunk = chunks[-1]
    return {
        "id": last_chunk.id,
        "object": "chat.completion",
        "model": last_chunk.model,
        "created": last_chunk.created,
        "choices": [choice.to_dict(index) for index, choice in sorted(choices.items())],
    }


class _StreamedChoice:
    """The message of one choice, accumulated from its stream deltas."""

    def __init__(self):
        self.role = "assistant"
        self.finish_reason = None
        self.content_parts: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        # The latest tool call of each index, which a fragment without a new ID continues.
        self.tool_call_by_index: dict[int, dict[str, Any]] = {}

    def add(self, choice) -> None:
        self.finish_reason = choice.finish_reason or self.finish_reason
        delta = choice.delta
        if isinstance(delta.role, str):
            self.role = delta.role
        _merge_content(self.content_parts, delta.content)
        for tool_call in delta.tool_calls or []:
            self._merge_tool_call(tool_call)

    def to_dict(self, index: int) -> dict[str, Any]:
        if all(part.get("type") == "text" for part in self.content_parts):
            content = "".join(part.get("text") or "" for part in self.content_parts)
        else:
            # Reasoning models stream thinking and text chunks, so keep them as a chunk list.
            content = self.content_parts
        message = {"role": self.role, "content": content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return {"index": index, "message": message, "finish_reason": self.finish_reason}

    def _merge_tool_call(self, tool_call) -> None:
        # Mistral usually sends each tool call whole in one chunk. A fragment continues the
        # latest call of the same index unless it is recognizably a new call.
        index = getattr(tool_call, "index", None) or 0
        call_id = tool_call.id if tool_call.id not in (None, "null") else None
        name = tool_call.function.name
        arguments = tool_call.function.arguments
        current = self.tool_call_by_index.get(index)
        if current is None or _starts_new_tool_call(current, call_id, name):
            new_call = tool_call.model_dump(exclude={"index"})
            self.tool_calls.append(new_call)
            self.tool_call_by_index[index] = new_call
            return

        function = current["function"]
        if isinstance(arguments, str) and arguments:
            existing = function.get("arguments") or ""
            if not isinstance(existing, str):
                existing = json.dumps(existing)
            function["arguments"] = existing + arguments
        if name and not function.get("name"):
            function["name"] = name
        if call_id and current.get("id") in (None, "null"):
            current["id"] = call_id


def _starts_new_tool_call(current: dict[str, Any], call_id: str | None, name: str | None) -> bool:
    current_id = current.get("id") if current.get("id") not in (None, "null") else None
    if call_id and current_id:
        return call_id != current_id
    # Without IDs to compare, a named fragment after a call whose arguments are already
    # complete is a new call (e.g. two calls without IDs that both default to index 0).
    function = current["function"]
    return bool(name and function.get("name") and _is_complete_json(function.get("arguments")))


def _is_complete_json(arguments: Any) -> bool:
    if isinstance(arguments, dict):
        return True
    if not isinstance(arguments, str):
        return False
    try:
        json.loads(arguments)
    except ValueError:
        return False
    return True


def _merge_content(parts: list[dict[str, Any]], content: Any) -> None:
    # ``content`` is either a string or a list of content chunks (e.g. text and thinking).
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content = [c.model_dump() if isinstance(c, BaseModel) else c for c in content]
    else:
        return

    for part in content:
        if not isinstance(part, dict):
            continue
        last = parts[-1] if parts else None
        match part, last:
            case {"type": "text", "text": str(text)}, {"type": "text"}:
                last["text"] = (last.get("text") or "") + text
            case {"type": "thinking"}, {"type": "thinking"} if not last.get("closed"):
                # A thinking block is streamed in fragments until a chunk closes it.
                _merge_thinking(last, part)
            case {"type": "thinking"}, _:
                # Older SDKs dump unset fields as None (e.g. ``closed``), so they are dropped.
                block = {k: v for k, v in part.items() if v is not None}
                parts.append({**block, "thinking": [dict(t) for t in part.get("thinking") or []]})
            case _:
                parts.append(dict(part))


def _merge_thinking(block: dict[str, Any], fragment: dict[str, Any]) -> None:
    items = block.setdefault("thinking", [])
    for item in fragment.get("thinking") or []:
        if item.get("type") == "text" and items and items[-1].get("type") == "text":
            items[-1]["text"] = (items[-1].get("text") or "") + (item.get("text") or "")
        else:
            items.append(dict(item))
    # Keep other fields (e.g. ``signature``, ``closed``) from the latest fragment that sets them.
    for key, value in fragment.items():
        if key not in ("type", "thinking") and value is not None:
            block[key] = value


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
