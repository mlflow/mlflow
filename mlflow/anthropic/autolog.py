import json
import logging
import weakref
from typing import Any

import mlflow.anthropic
from mlflow.anthropic.chat import convert_tool_to_mlflow_chat_tool
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import (
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
    SpanAttributeKey,
    TokenUsageKey,
)
from mlflow.tracing.distributed import _get_tracing_headers_from_span
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.utils import (
    TraceJSONEncoder,
    construct_full_inputs,
    set_span_chat_tools,
    set_span_model_attribute,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_claude_sdk_init(original, self, options=None):
    try:
        from claude_agent_sdk.types import UserMessage

        result = original(self, options)
        messages = []

        # query() sends the user prompt but doesn't echo it through receive_response()
        original_query = self.query

        async def wrapped_query(prompt, *args, **kwargs):
            if isinstance(prompt, str):
                messages.append(UserMessage(content=prompt))
            elif hasattr(prompt, "__aiter__"):
                # prompt is an async generator yielding message dicts — wrap it
                # to capture the user content while passing items through to the SDK
                original_prompt = prompt

                async def capturing_prompt():
                    async for item in original_prompt:
                        if isinstance(item, dict) and item.get("type") == "user":
                            content = item.get("message", {}).get("content", "")
                            if isinstance(content, str) and content.strip():
                                messages.append(UserMessage(content=content))
                        yield item

                prompt = capturing_prompt()
            return await original_query(prompt, *args, **kwargs)

        self.query = wrapped_query

        original_receive_response = self.receive_response

        async def wrapped_receive_response(*args, **kwargs):
            async for msg in original_receive_response(*args, **kwargs):
                messages.append(msg)
                yield msg
            try:
                from mlflow.utils.autologging_utils import autologging_is_disabled

                if not autologging_is_disabled("anthropic"):
                    from mlflow.claude_code.tracing import process_sdk_messages

                    process_sdk_messages(list(messages))
            except Exception as e:
                _logger.debug("Error building SDK trace: %s", e, exc_info=True)

        self.receive_response = wrapped_receive_response
        return result
    except Exception as e:
        _logger.debug("Error in patched_claude_sdk_init: %s", e, exc_info=True)
        return original(self, options)


def patched_class_call(original, self, *args, **kwargs):
    with TracingSession(original, self, args, kwargs) as manager:
        _inject_tracing_headers(kwargs, manager.span)
        output = original(self, *args, **kwargs)
        manager.set_output(output)
        return output


async def async_patched_class_call(original, self, *args, **kwargs):
    async with TracingSession(original, self, args, kwargs) as manager:
        _inject_tracing_headers(kwargs, manager.span)
        output = await original(self, *args, **kwargs)
        manager.set_output(output)
        return output


def patched_stream_call(original, self, *args, **kwargs):
    with TracingSession(original, self, args, kwargs) as session:
        _inject_tracing_headers(kwargs, session.span)
        stream_manager = original(self, *args, **kwargs)
        if session.span is None:
            return stream_manager
        # `stream()` only prepares the request, which is sent once the returned manager is
        # entered. The span stays open and goes to a stand-in that follows the stream from there.
        session.deferred = True
        if hasattr(stream_manager, "__aenter__"):
            return _TracedAsyncStreamManager(stream_manager, session)
        return _TracedStreamManager(stream_manager, session)


class TracingSession:
    """Context manager for handling MLflow spans in both sync and async contexts."""

    def __init__(self, original, instance, args, kwargs):
        # `original` is not kept: it holds on to what the call returned. When that is a stream,
        # the session lives as long as the stream does and would keep a dropped stream alive.
        self.name = f"{instance.__class__.__name__}.{original.__name__}"
        self.span_type = _get_span_type(original.__name__)
        self.inputs = construct_full_inputs(original, instance, *args, **kwargs)

        # These attributes are set outside the constructor.
        self.span = None
        self.output = None
        # Set when the response is streamed, so that the span outlives the patched call.
        self.deferred = False
        self.ended = False

    def __enter__(self):
        return self._enter_impl()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self._enter_impl()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    def _enter_impl(self):
        config = AutoLoggingConfig.init(flavor_name=mlflow.anthropic.FLAVOR_NAME)

        if config.log_traces:
            self.span = start_span_no_context(
                name=self.name,
                span_type=self.span_type,
                inputs=self.inputs,
                attributes={SpanAttributeKey.MESSAGE_FORMAT: "anthropic"},
            )
            _set_tool_attribute(self.span, self.inputs)

        return self

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        if self.span and not self.deferred:
            self.end(exc_val)

    def set_output(self, output: Any) -> None:
        from anthropic import AsyncStream, Stream

        if self.span and isinstance(output, (Stream, AsyncStream)):
            # `create(stream=True)` returns before any event is read.
            self.trace_stream(output)
        else:
            self.output = output

    def trace_stream(self, stream: Any) -> None:
        """Keep the span open until ``stream`` ends, then record the message it delivered."""
        _trace_stream(self, stream)
        self.deferred = True

    def end(self, error: BaseException | None = None) -> None:
        # A stream can end in several ways, e.g. it is read to the end and then closed.
        if self.ended:
            return
        self.ended = True

        if isinstance(error, Exception):
            self.span.record_exception(error)
        elif error is not None:
            # `record_exception` rejects errors that are not an `Exception`, such as the
            # `asyncio.CancelledError` of a stream that is cancelled.
            self.span.add_event(SpanEvent.from_exception(error))
            self.span.set_status(SpanStatusCode.ERROR)

        set_span_model_attribute(self.span, self.inputs)
        # Client-side cost computation (used for Databricks backends) resolves
        # litellm pricing by provider; without it, Claude model names don't
        # match and cost is silently dropped while token usage is still
        # recorded. This autolog patches the Anthropic SDK, so the provider
        # is always Anthropic.
        self.span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, "anthropic")
        _set_token_usage_attribute(self.span, self.output)
        self.span.end(outputs=self.output)


class _TracedStreamManager:
    """
    Stands in for the manager that ``Messages.stream()`` returns, to trace the stream it opens.
    """

    def __init__(self, manager: Any, session: TracingSession):
        self._manager = manager
        self._session = session

    def __enter__(self):
        try:
            stream = self._manager.__enter__()
        except BaseException as e:
            _end_span(self._session, e)
            raise
        _trace_helper_stream(self._session, stream)
        return stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._manager.__exit__(exc_type, exc_val, exc_tb)


class _TracedAsyncStreamManager:
    """
    Stands in for the manager that ``AsyncMessages.stream()`` returns, to trace the stream it
    opens.
    """

    def __init__(self, manager: Any, session: TracingSession):
        self._manager = manager
        self._session = session

    async def __aenter__(self):
        try:
            stream = await self._manager.__aenter__()
        except BaseException as e:
            _end_span(self._session, e)
            raise
        _trace_helper_stream(self._session, stream)
        return stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._manager.__aexit__(exc_type, exc_val, exc_tb)


class _StreamRecorder:
    """Records the events of a streamed response and ends the span once the stream is over."""

    def __init__(self, session: TracingSession):
        self._session = session
        self._events = []

    def iterate(self, iterator):
        try:
            for event in iterator:
                self._record(event)
                yield event
        except Exception as e:
            self.finish(e)
            raise
        finally:
            # Also reached when the caller stops reading and the generator is closed.
            self.finish()

    async def aiterate(self, iterator):
        try:
            async for event in iterator:
                self._record(event)
                yield event
        except GeneratorExit:
            raise
        except BaseException as e:
            # Includes `asyncio.CancelledError`, raised when the task reading the stream is
            # cancelled.
            self.finish(e)
            raise
        finally:
            self.finish()

    def _record(self, event: Any) -> None:
        try:
            self._session.span.add_event(
                SpanEvent(
                    name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=len(self._events)),
                    attributes={
                        STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(event, cls=TraceJSONEncoder)
                    },
                )
            )
        except Exception:
            _logger.debug("Failed to record an Anthropic stream event", exc_info=True)
        self._events.append(event)

    def finish(self, error: BaseException | None = None) -> None:
        if self._session.ended:
            return
        try:
            self._session.output = _accumulate_message(self._events)
        except Exception:
            _logger.warning("Failed to assemble the message of an Anthropic stream", exc_info=True)
        _end_span(self._session, error)


def _end_span(session: TracingSession, error: BaseException | None = None) -> None:
    # Streams end their span after the patched call has returned, where an error raised here
    # would surface in the caller's own code.
    try:
        session.end(error)
    except Exception:
        _logger.warning("Failed to end the span of an Anthropic stream", exc_info=True)


def _trace_stream(session: TracingSession, stream: Any) -> None:
    """
    Keep the span open until ``stream`` is read to the end, closed, dropped or fails, then end
    it with the message assembled from the streamed events. ``stream`` is the SDK's ``Stream``
    or ``AsyncStream``.
    """
    recorder = _StreamRecorder(session)
    # `close` is taken from the class and called through a weak reference. Keeping the bound
    # method instead would tie the stream into a reference cycle, and a stream that is dropped
    # without being closed would stay open, like its span, until the garbage collector runs.
    close = type(stream).close
    stream_ref = weakref.ref(stream)

    if hasattr(stream, "__aiter__"):

        async def traced_close():
            try:
                await close(stream_ref())
            except BaseException as e:
                recorder.finish(e)
                raise
            recorder.finish()

        stream._iterator = recorder.aiterate(stream._iterator)
    else:

        def traced_close():
            try:
                close(stream_ref())
            except BaseException as e:
                recorder.finish(e)
                raise
            recorder.finish()

        stream._iterator = recorder.iterate(stream._iterator)

    stream.close = traced_close
    # A stream can also be dropped before it is read to the end or closed.
    weakref.finalize(stream, recorder.finish)


def _trace_helper_stream(session: TracingSession, stream: Any) -> None:
    if session.ended:
        # The manager was entered before, and the span covered the stream it opened then.
        return
    try:
        # The stream that `stream()` opens reads its events from a raw stream, the kind that
        # `create(stream=True)` returns.
        session.trace_stream(stream._raw_stream)
    except Exception:
        _logger.warning("Failed to trace an Anthropic stream", exc_info=True)
        _end_span(session)


def _accumulate_message(events: list[Any]) -> Any:
    """
    Fold the events of a stream into the ``Message`` that a non-streaming call would have
    returned. A stream that stopped early gives the message as far as it was received.
    """
    message = None
    blocks = {}
    partial_inputs = {}
    for event in events:
        match event.type:
            case "message_start":
                message = event.message.model_copy(deep=True)
            case "content_block_start":
                blocks[event.index] = event.content_block.model_copy(deep=True)
            case "content_block_delta" if (block := blocks.get(event.index)) is not None:
                delta = event.delta
                match delta.type:
                    case "text_delta":
                        block.text += delta.text
                    case "thinking_delta":
                        block.thinking += delta.thinking
                    case "signature_delta":
                        block.signature = delta.signature
                    case "citations_delta":
                        block.citations = [*(block.citations or []), delta.citation]
                    case "input_json_delta":
                        partial_inputs[event.index] = (
                            partial_inputs.get(event.index, "") + delta.partial_json
                        )
            case "message_delta" if message is not None:
                _merge(message, event.delta)
                _merge(message.usage, event.usage)

    if message is None:
        return None

    for index, partial_json in partial_inputs.items():
        if partial_json:
            blocks[index].input = _parse_tool_input(partial_json, default=blocks[index].input)
    message.content = [blocks[index] for index in sorted(blocks)]
    return message


def _merge(target: Any, update: Any) -> None:
    # `message_delta` carries the final values but leaves out the ones that did not change,
    # e.g. the input token count. Those keep what `message_start` reported.
    for name, value in update:
        if value is not None and name in type(target).model_fields:
            setattr(target, name, value)


def _parse_tool_input(partial_json: str, default: Any) -> Any:
    # jiter is installed with the SDK, whose own stream helper parses tool inputs the same way.
    # The partial mode keeps what was received of an input that was cut off, e.g. by
    # `max_tokens` or because the stream was closed early.
    from jiter import from_json

    try:
        return from_json(partial_json.encode(), partial_mode=True)
    except ValueError:
        return default


def _inject_tracing_headers(kwargs: dict[str, Any], span: LiveSpan | None):
    if span is None:
        return
    try:
        if tracing_headers := _get_tracing_headers_from_span(span):
            existing = kwargs.get("extra_headers") or {}
            kwargs["extra_headers"] = tracing_headers | existing
    except Exception:
        _logger.debug("Failed to inject tracing headers", exc_info=True)


def _get_span_type(task_name: str) -> str:
    # Anthropic has a few APIs in beta, e.g., count_tokens.
    # Once they are stable, we can add them to the mapping.
    span_type_mapping = {
        "create": SpanType.CHAT_MODEL,
        "stream": SpanType.CHAT_MODEL,
    }
    return span_type_mapping.get(task_name, SpanType.UNKNOWN)


def _set_tool_attribute(span: LiveSpan, inputs: dict[str, Any]):
    if (tools := inputs.get("tools")) is not None:
        try:
            tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools]
            set_span_chat_tools(span, tools)
        except Exception as e:
            _logger.debug(f"Failed to set tools for {span}. Error: {e}")


def _set_token_usage_attribute(span: LiveSpan, output: Any):
    try:
        if usage := _parse_usage(output):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
    except Exception as e:
        _logger.debug(f"Failed to set token usage for {span}. Error: {e}")


def _parse_usage(output: Any) -> dict[str, int] | None:
    try:
        if usage := getattr(output, "usage", None):
            usage_dict = {
                TokenUsageKey.INPUT_TOKENS: usage.input_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.output_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.input_tokens + usage.output_tokens,
            }
            if (cached := getattr(usage, "cache_read_input_tokens", None)) is not None:
                usage_dict[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = cached
            if (created := getattr(usage, "cache_creation_input_tokens", None)) is not None:
                usage_dict[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = created
            # cache_creation carries a per-TTL breakdown; capture the 1-hour portion (a subset
            # of cache_creation_input_tokens) so cost can price it at the 1-hour rate.
            if (cache_creation_details := getattr(usage, "cache_creation", None)) is not None:
                if (
                    above_1hr := getattr(cache_creation_details, "ephemeral_1h_input_tokens", None)
                ) is not None:
                    usage_dict[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR] = above_1hr
            # Anthropic reports input_tokens excluding cache tokens. Normalize to
            # include them, consistent with OpenAI/Gemini and cost_per_token().
            # Same logic as _normalize_anthropic_input_tokens in gateway/providers/anthropic.py.
            if cache_total := (cached or 0) + (created or 0):
                usage_dict[TokenUsageKey.INPUT_TOKENS] += cache_total
                usage_dict[TokenUsageKey.TOTAL_TOKENS] += cache_total
            return usage_dict
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from output: {e}")
    return None
