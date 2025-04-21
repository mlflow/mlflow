import functools
import json
import logging
from typing import Any, AsyncIterator, Iterator

from packaging.version import Version

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.openai.utils.chat_schema import set_span_chat_attributes
from mlflow.tracing.assessment import MlflowClient
from mlflow.tracing.constant import (
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
    TraceMetadataKey,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    TraceJSONEncoder,
    end_client_span_or_trace,
    start_client_span_or_trace,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

MIN_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["openai"]["autologging"]["minimum"])
MAX_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["openai"]["autologging"]["maximum"])

_logger = logging.getLogger(__name__)


def _get_span_type(task: type) -> str:
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
        pass

    try:
        # Responses API only available in openai>=1.66.0
        from openai.resources.responses import AsyncResponses, Responses

        span_type_mapping[Responses] = SpanType.CHAT_MODEL
        span_type_mapping[AsyncResponses] = SpanType.CHAT_MODEL
    except ImportError:
        pass

    return span_type_mapping.get(task, SpanType.UNKNOWN)


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


def patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.openai.FLAVOR_NAME)
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else None
    mlflow_client = mlflow.MlflowClient()

    if config.log_traces:
        span = _start_span(mlflow_client, self, kwargs, run_id)

    # Execute the original function
    try:
        raw_result = original(self, *args, **kwargs)
    except Exception as e:
        if config.log_traces:
            _end_span_on_exception(mlflow_client, span, e)
        raise

    if config.log_traces:
        _end_span_on_success(mlflow_client, span, kwargs, raw_result)

    return raw_result


async def async_patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.openai.FLAVOR_NAME)
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else None
    mlflow_client = mlflow.MlflowClient()

    if config.log_traces:
        span = _start_span(mlflow_client, self, kwargs, run_id)

    # Execute the original function
    try:
        raw_result = await original(self, *args, **kwargs)
    except Exception as e:
        if config.log_traces:
            _end_span_on_exception(mlflow_client, span, e)
        raise

    if config.log_traces:
        _end_span_on_success(mlflow_client, span, kwargs, raw_result)

    return raw_result


def _start_span(
    mlflow_client: MlflowClient,
    instance: Any,
    inputs: dict[str, Any],
    run_id: str,
):
    # Record input parameters to attributes
    attributes = {k: v for k, v in inputs.items() if k not in ("messages", "input")}

    # If there is an active span, create a child span under it, otherwise create a new trace
    span = start_client_span_or_trace(
        mlflow_client,
        name=instance.__class__.__name__,
        span_type=_get_span_type(instance.__class__),
        inputs=inputs,
        attributes=attributes,
    )

    # Associate run ID to the trace manually, because if a new run is created by
    # autologging, it is not set as the active run thus not automatically
    # associated with the trace.
    if run_id is not None:
        tm = InMemoryTraceManager().get_instance()
        tm.set_request_metadata(span.trace_id, TraceMetadataKey.SOURCE_RUN, run_id)

    return span


def _end_span_on_success(
    mlflow_client: MlflowClient, span: LiveSpan, inputs: dict[str, Any], raw_result: Any
):
    from openai import AsyncStream, Stream

    result = _try_parse_raw_response(raw_result)

    if isinstance(result, Stream):
        # If the output is a stream, we add a hook to store the intermediate chunks
        # and then log the outputs as a single artifact when the stream ends
        def _stream_output_logging_hook(stream: Iterator) -> Iterator:
            output = []
            for i, chunk in enumerate(stream):
                output.append(_process_chunk(span, i, chunk))
                yield chunk
            output = chunk.response if _is_responses_final_event(chunk) else "".join(output)
            _end_span_on_success(mlflow_client, span, inputs, output)

        result._iterator = _stream_output_logging_hook(result._iterator)
    elif isinstance(result, AsyncStream):

        async def _stream_output_logging_hook(stream: AsyncIterator) -> AsyncIterator:
            output = []
            async for chunk in stream:
                output.append(_process_chunk(span, len(output), chunk))
                yield chunk
            output = chunk.response if _is_responses_final_event(chunk) else "".join(output)
            _end_span_on_success(mlflow_client, span, inputs, output)

        result._iterator = _stream_output_logging_hook(result._iterator)
    else:
        try:
            set_span_chat_attributes(span, inputs, result)
            end_client_span_or_trace(mlflow_client, span, outputs=result)
        except Exception as e:
            _logger.warning(f"Encountered unexpected error when ending trace: {e}", exc_info=True)


def _is_responses_final_event(chunk: Any) -> bool:
    try:
        from openai.types.responses import ResponseCompletedEvent

        return isinstance(chunk, ResponseCompletedEvent)
    except ImportError:
        return False


def _end_span_on_exception(mlflow_client: MlflowClient, span: LiveSpan, e: Exception):
    try:
        span.add_event(SpanEvent.from_exception(e))
        mlflow_client.end_span(span.trace_id, span.span_id, status=SpanStatusCode.ERROR)
    except Exception as inner_e:
        _logger.warning(f"Encountered unexpected error when ending trace: {inner_e}")


def _process_chunk(span: LiveSpan, index: int, chunk: Any) -> str:
    """Parse the chunk and log it as a span event in the trace."""
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.completion import Completion

    # `chunk.choices` can be empty: https://github.com/mlflow/mlflow/issues/13361
    if isinstance(chunk, Completion) and chunk.choices:
        parsed = chunk.choices[0].text or ""
    elif isinstance(chunk, ChatCompletionChunk) and chunk.choices:
        choice = chunk.choices[0]
        parsed = (choice.delta and choice.delta.content) or ""
    else:
        parsed = ""

    span.add_event(
        SpanEvent(
            name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=index),
            # OpenTelemetry SpanEvent only support str-str key-value pairs for attributes
            attributes={STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(chunk, cls=TraceJSONEncoder)},
        )
    )
    return parsed


def patched_agent_get_chat_completion(original, self, *args, **kwargs):
    """
    Patch the `get_chat_completion` method of the ChatCompletion object.
    OpenAI autolog already handles the raw completion request, but tracing
    the swarm's method is useful to track other parameters like agent name.
    """
    agent = kwargs.get("agent") or args[0]

    # Patch agent's functions to generate traces. Function calls only happen
    # after the first completion is generated because of the design of
    # function calling. Therefore, we can safely patch the tool functions here
    # within get_chat_completion() hook.
    # We cannot patch functions during the agent's initialization because the
    # agent's functions can be modified after the agent is created.
    def function_wrapper(fn):
        if "context_variables" in fn.__code__.co_varnames:

            def wrapper(*args, **kwargs):
                # NB: Swarm uses `func.__code__.co_varnames` to inspect if the provided
                # tool function includes 'context_variables' parameter in the signature
                # and ingest the global context variables if so. Wrapping the function
                # with mlflow.trace() will break this.
                # The co_varnames is determined based on the local variables of the
                # function, so we workaround this by declaring it here as a local variable.
                context_variables = kwargs.get("context_variables", {})  # noqa: F841
                return mlflow.trace(
                    fn,
                    name=f"{agent.name}.{fn.__name__}",
                    span_type=SpanType.TOOL,
                )(*args, **kwargs)
        else:

            def wrapper(*args, **kwargs):
                return mlflow.trace(
                    fn,
                    name=f"{agent.name}.{fn.__name__}",
                    span_type=SpanType.TOOL,
                )(*args, **kwargs)

        wrapped = functools.wraps(fn)(wrapper)
        wrapped._is_mlflow_traced = True  # Marker to avoid double tracing
        return wrapped

    agent.functions = [
        function_wrapper(fn) if not hasattr(fn, "_is_mlflow_traced") else fn
        for fn in agent.functions
    ]

    traced_fn = mlflow.trace(
        original, name=f"{agent.name}.get_chat_completion", span_type=SpanType.CHAIN
    )
    return traced_fn(self, *args, **kwargs)


def patched_swarm_run(original, self, *args, **kwargs):
    """
    Patched version of `run` method of the Swarm object.
    """
    traced_fn = mlflow.trace(original, span_type=SpanType.AGENT)
    return traced_fn(self, *args, **kwargs)
