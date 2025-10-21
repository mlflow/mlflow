import importlib.metadata
import json
import logging
from typing import Any, AsyncIterator, Iterator

from packaging.version import Version

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.exceptions import MlflowException
from mlflow.openai.constant import FLAVOR_NAME
from mlflow.openai.utils.chat_schema import set_span_chat_attributes
from mlflow.tracing.constant import (
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
    SpanAttributeKey,
    TokenUsageKey,
    TraceMetadataKey,
)
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import safe_patch

_logger = logging.getLogger(__name__)


def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    """
    Enables (or disables) and configures autologging from OpenAI to MLflow.
    Raises :py:class:`MlflowException <mlflow.exceptions.MlflowException>`
    if the OpenAI version < 1.0.

    Args:
        disable: If ``True``, disables the OpenAI autologging integration. If ``False``,
            enables the OpenAI autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            OpenAI that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during OpenAI
            autologging. If ``False``, show all events and warnings during OpenAI
            autologging.
        log_traces: If ``True``, traces are logged for OpenAI models. If ``False``, no traces are
            collected during inference. Default to ``True``.
    """
    if Version(importlib.metadata.version("openai")).major < 1:
        raise MlflowException("OpenAI autologging is only supported for openai >= 1.0.0")

    # This needs to be called before doing any safe-patching (otherwise safe-patch will be no-op).
    # TODO: since this implementation is inconsistent, explore a universal way to solve the issue.
    _autolog(
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        log_traces=log_traces,
    )

    # Tracing OpenAI Agent SDK. This has to be done outside the function annotated with
    # `@autologging_integration` because the function is not executed when `disable=True`.
    try:
        from agents.run import AgentRunner

        from mlflow.openai._agent_tracer import _patched_agent_run

        # NB: The OpenAI's built-in tracer does not capture inputs/outputs of the
        # root span, which is not inconvenient. Therefore, we add a patch for the
        # runner.run() method instead.
        safe_patch(FLAVOR_NAME, AgentRunner, "run", _patched_agent_run)

        from mlflow.openai._agent_tracer import (
            add_mlflow_trace_processor,
            remove_mlflow_trace_processor,
        )

        if log_traces and not disable:
            add_mlflow_trace_processor()
        else:
            remove_mlflow_trace_processor()
    except ImportError:
        pass


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


# NB: The @autologging_integration annotation must be applied here, and the callback injection
# needs to happen outside the annotated function. This is because the annotated function is NOT
# executed when disable=True is passed. This prevents us from removing our callback and patching
# when autologging is turned off.
@autologging_integration(FLAVOR_NAME)
def _autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.chat.completions import Completions as ChatCompletions
    from openai.resources.completions import AsyncCompletions, Completions
    from openai.resources.embeddings import AsyncEmbeddings, Embeddings

    for task in (ChatCompletions, Completions, Embeddings):
        safe_patch(FLAVOR_NAME, task, "create", patched_call)

    if hasattr(ChatCompletions, "parse"):
        # In openai>=1.92.0, `ChatCompletions` has a `parse` method:
        # https://github.com/openai/openai-python/commit/0e358ed66b317038705fb38958a449d284f3cb88
        safe_patch(FLAVOR_NAME, ChatCompletions, "parse", patched_call)

    for task in (AsyncChatCompletions, AsyncCompletions, AsyncEmbeddings):
        safe_patch(FLAVOR_NAME, task, "create", async_patched_call)

    if hasattr(AsyncChatCompletions, "parse"):
        # In openai>=1.92.0, `AsyncChatCompletions` has a `parse` method:
        # https://github.com/openai/openai-python/commit/0e358ed66b317038705fb38958a449d284f3cb88
        safe_patch(FLAVOR_NAME, AsyncChatCompletions, "parse", async_patched_call)

    try:
        from openai.resources.beta.chat.completions import AsyncCompletions, Completions
    except ImportError:
        pass
    else:
        safe_patch(FLAVOR_NAME, Completions, "parse", patched_call)
        safe_patch(FLAVOR_NAME, AsyncCompletions, "parse", async_patched_call)

    try:
        from openai.resources.responses import AsyncResponses, Responses
    except ImportError:
        pass
    else:
        safe_patch(FLAVOR_NAME, Responses, "create", patched_call)
        safe_patch(FLAVOR_NAME, AsyncResponses, "create", async_patched_call)
        safe_patch(FLAVOR_NAME, AsyncResponses, "parse", async_patched_call)
        safe_patch(FLAVOR_NAME, Responses, "parse", patched_call)


def _get_span_type_and_message_format(task: type) -> tuple[str, str]:
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.chat.completions import Completions as ChatCompletions
    from openai.resources.completions import AsyncCompletions, Completions
    from openai.resources.embeddings import AsyncEmbeddings, Embeddings

    span_type_mapping = {
        ChatCompletions: SpanType.CHAT_MODEL,
        AsyncChatCompletions: SpanType.CHAT_MODEL,
        Completions: SpanType.LLM,
        AsyncCompletions: SpanType.LLM,
        Embeddings: SpanType.EMBEDDING,
        AsyncEmbeddings: SpanType.EMBEDDING,
    }

    try:
        # Only available in openai>=1.40.0
        from openai.resources.beta.chat.completions import (
            AsyncCompletions as BetaAsyncChatCompletions,
        )
        from openai.resources.beta.chat.completions import Completions as BetaChatCompletions

        span_type_mapping[BetaChatCompletions] = SpanType.CHAT_MODEL
        span_type_mapping[BetaAsyncChatCompletions] = SpanType.CHAT_MODEL
    except ImportError:
        _logger.debug(
            "Failed to import `BetaChatCompletions` or `BetaAsyncChatCompletions`", exc_info=True
        )

    try:
        # Responses API only available in openai>=1.66.0
        from openai.resources.responses import AsyncResponses, Responses

        span_type_mapping[Responses] = SpanType.CHAT_MODEL
        span_type_mapping[AsyncResponses] = SpanType.CHAT_MODEL
    except ImportError:
        pass

    return span_type_mapping.get(task, (SpanType.UNKNOWN, None))


def _try_parse_raw_response(response: Any) -> Any:
    """
    As documented at https://github.com/openai/openai-python/tree/52357cff50bee57ef442e94d78a0de38b4173fc2?tab=readme-ov-file#accessing-raw-response-data-eg-headers,
    a `LegacyAPIResponse` (https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/_legacy_response.py#L45)
    object is returned when the `create` method is invoked with `with_raw_response`.
    """
    try:
        from openai._legacy_response import LegacyAPIResponse
    except ImportError:
        _logger.debug("Failed to import `LegacyAPIResponse` from `openai._legacy_response`")
        return response
    if isinstance(response, LegacyAPIResponse):
        try:
            # `parse` returns either a `pydantic.BaseModel` or a `openai.Stream` object
            # depending on whether the request has a `stream` parameter set to `True`.
            return response.parse()
        except Exception as e:
            _logger.debug(f"Failed to parse {response} (type: {response.__class__}): {e}")

    return response


def _is_responses_api(original: Any) -> bool:
    match getattr(original, "__qualname__", "").split("."):
        case [class_name, _]:
            return class_name in {"Responses", "AsyncResponses"}
        case _:
            return False


def patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.openai.FLAVOR_NAME)
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else None

    if config.log_traces:
        span = _start_span(self, kwargs, run_id)

    # Execute the original function
    try:
        raw_result = original(self, *args, **kwargs)
    except Exception as e:
        if config.log_traces:
            _end_span_on_exception(span, e)
        raise

    if config.log_traces:
        _end_span_on_success(span, kwargs, raw_result, is_responses_api=_is_responses_api(original))

    return raw_result


async def async_patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.openai.FLAVOR_NAME)
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else None

    if config.log_traces:
        span = _start_span(self, kwargs, run_id)

    # Execute the original function
    try:
        raw_result = await original(self, *args, **kwargs)
    except Exception as e:
        if config.log_traces:
            _end_span_on_exception(span, e)
        raise

    if config.log_traces:
        _end_span_on_success(span, kwargs, raw_result, is_responses_api=_is_responses_api(original))

    return raw_result


def _start_span(
    instance: Any,
    inputs: dict[str, Any],
    run_id: str,
):
    span_type = _get_span_type_and_message_format(instance.__class__)
    # Record input parameters to attributes
    attributes = {k: v for k, v in inputs.items() if k not in ("messages", "input")}
    if span_type in (SpanType.CHAT_MODEL, SpanType.LLM):
        attributes[SpanAttributeKey.MESSAGE_FORMAT] = "openai"

    # If there is an active span, create a child span under it, otherwise create a new trace
    span = start_span_no_context(
        name=instance.__class__.__name__,
        span_type=span_type,
        inputs=inputs,
        attributes=attributes,
    )

    # Associate run ID to the trace manually, because if a new run is created by
    # autologging, it is not set as the active run thus not automatically
    # associated with the trace.
    if run_id is not None:
        tm = InMemoryTraceManager().get_instance()
        tm.set_trace_metadata(span.trace_id, TraceMetadataKey.SOURCE_RUN, run_id)

    return span


def _end_span_on_success(
    span: LiveSpan,
    inputs: dict[str, Any],
    raw_result: Any,
    is_responses_api: bool,
):
    from openai import AsyncStream, Stream

    result = _try_parse_raw_response(raw_result)

    if isinstance(result, Stream):
        # If the output is a stream, we add a hook to store the intermediate chunks
        # and then log the outputs as a single artifact when the stream ends
        def _stream_output_logging_hook(stream: Iterator) -> Iterator:
            output = []
            for i, chunk in enumerate(stream):
                _add_span_event(span, i, chunk)
                output.append(chunk)
                yield chunk
            _process_last_chunk(span, chunk, inputs, output, is_responses_api)

        result._iterator = _stream_output_logging_hook(result._iterator)
    elif isinstance(result, AsyncStream):

        async def _stream_output_logging_hook(stream: AsyncIterator) -> AsyncIterator:
            output = []
            async for chunk in stream:
                _add_span_event(span, len(output), chunk)
                output.append(chunk)
                yield chunk
            _process_last_chunk(span, chunk, inputs, output, is_responses_api)

        result._iterator = _stream_output_logging_hook(result._iterator)
    else:
        try:
            set_span_chat_attributes(span, inputs, result)
            span.end(outputs=result)
        except Exception as e:
            _logger.warning(f"Encountered unexpected error when ending trace: {e}", exc_info=True)


def _process_last_chunk(
    span: LiveSpan,
    chunk: Any,
    inputs: dict[str, Any],
    output: list[Any],
    is_responses_api: bool,
) -> None:
    try:
        if _is_responses_final_event(chunk):
            output = chunk.response
        elif not output:
            output = None
        elif is_responses_api:
            output = _reconstruct_response_from_stream(output)
        elif output[0].object in ["text_completion", "chat.completion.chunk"]:
            # Reconstruct a completion object from streaming chunks
            output = _reconstruct_completion_from_stream(output)
            # Set usage information on span if available
            if usage := getattr(chunk, "usage", None):
                usage_dict = {
                    TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
                    TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
                    TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
                }
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)

        _end_span_on_success(span, inputs, output, is_responses_api)
    except Exception as e:
        _logger.warning(
            f"Encountered unexpected error when autologging processes the chunks in response: {e}"
        )


def _reconstruct_completion_from_stream(chunks: list[Any]) -> Any:
    """
    Reconstruct a completion object from streaming chunks.

    This preserves the structure and metadata that would be present in a non-streaming
    completion response, including ID, model, timestamps, usage, etc.
    """
    if chunks[0].object == "text_completion":
        # Handling for the deprecated Completions API. Keep the legacy behavior for now.
        def _extract_content(chunk: Any) -> str:
            if not chunk.choices:
                return ""
            return chunk.choices[0].text or ""

        return "".join(map(_extract_content, chunks))

    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    # Build the base message
    def _extract_content(chunk: Any) -> str:
        if not chunk.choices:
            return ""
        content = chunk.choices[0].delta.content
        if content is None:
            return ""
        # Handle Databricks streaming format where content can be a list of content items
        # See https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference#content-item
        if isinstance(content, list):
            text_parts = []
            for item in content:
                # Extract text from text items only.
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
            return "".join(text_parts)
        return content

    message = ChatCompletionMessage(
        role="assistant", content="".join(map(_extract_content, chunks))
    )

    # Extract metadata from the last chunk
    last_chunk = chunks[-1]
    finish_reason = "stop"
    if choices := getattr(last_chunk, "choices", None):
        if chunk_choice := choices[0]:
            finish_reason = getattr(chunk_choice, "finish_reason") or finish_reason

    choice = Choice(index=0, message=message, finish_reason=finish_reason)

    # Build the completion dict
    return ChatCompletion(
        id=last_chunk.id,
        choices=[choice],
        created=last_chunk.created,
        model=last_chunk.model,
        object="chat.completion",
        system_fingerprint=last_chunk.system_fingerprint,
        usage=last_chunk.usage,
    )


def _reconstruct_response_from_stream(chunks: list[Any]) -> Any:
    from openai.types.responses import ResponseOutputItemDoneEvent

    from mlflow.types.responses_helpers import Response

    output = [
        chunk.item.to_dict() for chunk in chunks if isinstance(chunk, ResponseOutputItemDoneEvent)
    ]

    return Response(output=output)


def _is_responses_final_event(chunk: Any) -> bool:
    try:
        from openai.types.responses import ResponseCompletedEvent

        return isinstance(chunk, ResponseCompletedEvent)
    except ImportError:
        return False


def _is_response_output_item_done_event(chunk: Any) -> bool:
    try:
        from openai.types.responses import ResponseOutputItemDoneEvent

        return isinstance(chunk, ResponseOutputItemDoneEvent)
    except ImportError:
        return False


def _end_span_on_exception(span: LiveSpan, e: Exception):
    try:
        span.add_event(SpanEvent.from_exception(e))
        span.end(status=SpanStatusCode.ERROR)
    except Exception as inner_e:
        _logger.warning(f"Encountered unexpected error when ending trace: {inner_e}")


def _add_span_event(span: LiveSpan, index: int, chunk: Any):
    span.add_event(
        SpanEvent(
            name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=index),
            # OpenTelemetry SpanEvent only support str-str key-value pairs for attributes
            attributes={STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(chunk, cls=TraceJSONEncoder)},
        )
    )
