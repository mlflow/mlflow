import contextlib
import contextvars
import json
import logging
import weakref
from typing import Any

from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.tracing.constant import (
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
)
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.utils import TraceJSONEncoder

_logger = logging.getLogger(__name__)
_active_stream = contextvars.ContextVar("gemini_active_stream", default=None)


class _StreamState:
    """
    Bookkeeping for one wrapped stream. It is kept apart from the wrapper and never references
    it, so dropping the wrapper frees it right away and its finalizer can end the span.
    """

    def __init__(self, stream, span: LiveSpan, on_finish):
        self.stream = stream
        self.span = span
        self.on_finish = on_finish
        self.chunks = []
        self.finished = False
        self.children = weakref.WeakSet()
        self.parent = _active_stream.get()
        if self.parent is not None:
            self.parent.children.add(self)

    def record_chunk(self, chunk):
        self.chunks.append(chunk)
        try:
            event = SpanEvent(
                name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=len(self.chunks) - 1),
                attributes={STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(chunk, cls=TraceJSONEncoder)},
            )
            self.span.add_event(event)
        except Exception:
            # Tracing must never break the caller's stream.
            _logger.warning("Failed to record Gemini streaming chunk", exc_info=True)

    def finish_children(self):
        for child in list(self.children):
            child.finish()

    def finish(self, error: BaseException | None = None):
        if self.finished:
            return
        self.finished = True
        # Nested streams (e.g. `Chat.send_message_stream` -> `Models.generate_content_stream`)
        # must end their spans before the parent's, otherwise the trace is exported incomplete.
        self.finish_children()
        if self.parent is not None:
            self.parent.children.discard(self)
            # The parent's suspended generator holds this stream's wrapper. Keeping the reference
            # would let this stream's finalizer keep both streams alive forever once abandoned.
            self.parent = None
        try:
            self.on_finish(self.chunks, error)
        except Exception:
            _logger.warning("Failed to end Gemini streaming span", exc_info=True)


@contextlib.contextmanager
def _stream_context(state: _StreamState):
    # Make the span the active one while the underlying stream produces a chunk, so spans
    # created by the SDK internals (e.g. the private `_generate_content_stream`) nest under it.
    token = set_span_in_context(state.span)
    stream_token = _active_stream.set(state)
    try:
        yield
    finally:
        _active_stream.reset(stream_token)
        detach_span_from_context(token)


class BaseStreamWrapper:
    def __init__(self, stream, span: LiveSpan, on_finish):
        self._state = _StreamState(stream, span, on_finish)
        # Callers often stop reading early (e.g. `break` out of the loop) without calling
        # `close()`. End the span once the wrapper is garbage-collected so the trace is still
        # exported. `finish` is idempotent, so this is a no-op for streams that already ended.
        weakref.finalize(self, self._state.finish)

    def __getattr__(self, attr):
        return getattr(self._state.stream, attr)


class SyncStreamWrapper(BaseStreamWrapper):
    def __iter__(self):
        return self

    def __next__(self):
        state = self._state
        try:
            with _stream_context(state):
                chunk = next(state.stream)
        except StopIteration:
            state.finish()
            raise
        except Exception as e:
            state.finish(e)
            raise

        state.record_chunk(chunk)
        return chunk

    def close(self):
        state = self._state
        state.finish_children()
        if close := getattr(state.stream, "close", None):
            try:
                close()
            except Exception as e:
                state.finish(e)
                raise
        state.finish()


class AsyncStreamWrapper(BaseStreamWrapper):
    def __aiter__(self):
        return self

    async def __anext__(self):
        state = self._state
        try:
            with _stream_context(state):
                chunk = await state.stream.__anext__()
        except StopAsyncIteration:
            state.finish()
            raise
        # `BaseException` covers `asyncio.CancelledError`, raised when the caller's task is
        # cancelled mid-stream (e.g. a client disconnects from a streaming endpoint).
        except BaseException as e:
            state.finish(e)
            raise

        state.record_chunk(chunk)
        return chunk

    async def aclose(self):
        state = self._state
        state.finish_children()
        if aclose := getattr(state.stream, "aclose", None):
            try:
                await aclose()
            except BaseException as e:
                state.finish(e)
                raise
        state.finish()


_TEXT_PART_FIELDS = {"text", "thought", "thought_signature"}


def construct_streaming_response(chunks: list[Any]):
    """
    Merge streamed `GenerateContentResponse` chunks into a single response. Metadata such as
    usage and finish reason comes from the last chunk, which carries the final values.
    """
    if not chunks:
        return None

    response = chunks[-1].model_copy(deep=True)
    candidates_by_index = {}
    parts_by_index = {}
    for chunk in chunks:
        for candidate in chunk.candidates or []:
            index = candidate.index or 0
            previous = candidates_by_index.get(index)
            current = candidate.model_copy(deep=True)
            if current.content is None and previous is not None:
                current.content = previous.content
            candidates_by_index[index] = current
            if candidate.content is not None and candidate.content.parts:
                parts_by_index.setdefault(index, []).extend(candidate.content.parts)

    if candidates_by_index:
        for index, candidate in candidates_by_index.items():
            if candidate.content is not None and index in parts_by_index:
                candidate.content = candidate.content.model_copy(
                    update={"parts": _merge_text_parts(parts_by_index[index])}
                )
        response.candidates = [candidates_by_index[index] for index in sorted(candidates_by_index)]
    return response


def _merge_text_parts(parts: list[Any]) -> list[Any]:
    # Gemini streams text as many small parts, and the last one is often an empty text part
    # that only carries the `thought_signature`. Join consecutive text parts so the span output
    # matches a non-streaming response. Thought and answer text are kept apart, and non-text
    # parts (e.g. function calls) are never merged.
    merged = []
    for part in parts:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and _is_text_part(previous)
            and _is_text_part(part)
            and bool(previous.thought) == bool(part.thought)
            and not (previous.thought_signature and part.thought_signature)
        ):
            merged[-1] = previous.model_copy(
                update={
                    "text": previous.text + part.text,
                    "thought_signature": previous.thought_signature or part.thought_signature,
                }
            )
        else:
            merged.append(part)
    return merged


def _is_text_part(part: Any) -> bool:
    return part.text is not None and all(
        value is None for name, value in part if name not in _TEXT_PART_FIELDS
    )
