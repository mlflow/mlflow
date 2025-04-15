import functools
import json
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, AsyncIterator, Iterator, Optional

from packaging.version import Version

import mlflow
from mlflow import MlflowException
from mlflow.entities import RunTag, SpanType
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
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.autologging_utils import disable_autologging, get_autologging_config
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

MIN_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["openai"]["autologging"]["minimum"])
MAX_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["openai"]["autologging"]["maximum"])

_logger = logging.getLogger(__name__)


def _get_input_from_model(model, kwargs):
    from openai.resources.chat.completions import Completions as ChatCompletions
    from openai.resources.completions import Completions
    from openai.resources.embeddings import Embeddings

    model_class_param_name_mapping = {
        ChatCompletions: "messages",
        Completions: "prompt",
        Embeddings: "input",
    }
    if param_name := model_class_param_name_mapping.get(model.__class__):
        # openai tasks accept only keyword arguments
        if param := kwargs.get(param_name):
            return param
        input_example_exc = MlflowException(
            "Inference function signature changes, please contact MLflow team to "
            "fix OpenAI autologging.",
        )
    else:
        input_example_exc = MlflowException(
            "Unsupported OpenAI task. Only support chat completions, completions and embeddings."
        )
    _logger.warning(
        f"Failed to gather input example of model {model.__class__.__name__} "
        f"due to error: {input_example_exc}"
    )


@contextmanager
def _set_api_key_env_var(client):
    """
    Gets the API key from the client and temporarily set it as an environment variable
    """
    api_key = client.api_key
    original = os.environ.get("OPENAI_API_KEY", None)
    os.environ["OPENAI_API_KEY"] = api_key
    yield
    if original is not None:
        os.environ["OPENAI_API_KEY"] = original
    else:
        os.environ.pop("OPENAI_API_KEY")


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
    run_id = _get_autolog_run_id(self, active_run)
    mlflow_client = mlflow.MlflowClient()

    # If optional artifacts logging are enabled e.g. log_models, we need to create a run
    if config.should_log_optional_artifacts():
        run_id = _start_run_or_log_tag(mlflow_client, config, run_id)

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

    if config.should_log_optional_artifacts():
        _log_optional_artifacts(config, run_id, self, kwargs)

    # Even if the model is not logged, we keep a single run per model
    self._mlflow_run_id = run_id

    # Terminate the run if it is not managed by the user
    if run_id is not None and (active_run is None or active_run.info.run_id != run_id):
        mlflow_client.set_terminated(run_id)

    return raw_result


async def async_patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.openai.FLAVOR_NAME)
    active_run = mlflow.active_run()
    run_id = _get_autolog_run_id(self, active_run)
    mlflow_client = mlflow.MlflowClient()

    # If optional artifacts logging are enabled e.g. log_models, we need to create a run
    if config.should_log_optional_artifacts():
        run_id = _start_run_or_log_tag(mlflow_client, config, run_id)

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

    if config.should_log_optional_artifacts():
        _log_optional_artifacts(config, run_id, self, kwargs)

    # Even if the model is not logged, we keep a single run per model
    self._mlflow_run_id = run_id

    # Terminate the run if it is not managed by the user
    if run_id is not None and (active_run is None or active_run.info.run_id != run_id):
        mlflow_client.set_terminated(run_id)

    return raw_result


def _get_autolog_run_id(instance, active_run):
    """
    Get the run ID to use for logging artifacts and associate with the trace.

    The run ID is determined as follows:
    - If there is an active run (created by a user), use its run ID.
    - If the model has a `_mlflow_run_id` attribute, use it. This is the run ID created
        by autologging in a previous call to the same model.
    """
    return active_run.info.run_id if active_run else getattr(instance, "_mlflow_run_id", None)


def _start_run_or_log_tag(
    mlflow_client: MlflowClient, config: AutoLoggingConfig, run_id: Optional[str]
) -> str:
    """Start a new run or log models, or log extra tags if a run is already active."""
    # include run context tags
    resolved_tags = context_registry.resolve_tags(config.extra_tags)
    tags = _resolve_extra_tags(mlflow.openai.FLAVOR_NAME, resolved_tags)
    if run_id is not None:
        mlflow_client.log_batch(
            run_id=run_id,
            tags=[RunTag(key, str(value)) for key, value in tags.items()],
        )
    else:
        run = mlflow_client.create_run(
            experiment_id=_get_experiment_id(),
            tags=tags,
        )
        run_id = run.info.run_id
    return run_id


def _log_optional_artifacts(
    config: AutoLoggingConfig, run_id: str, instance: Any, kwargs: dict[str, Any]
):
    if hasattr(instance, "_mlflow_model_logged"):
        # Model is already logged for this instance, no need to log again
        return

    input_example = None
    if config.log_input_examples:
        input_example = deepcopy(_get_input_from_model(instance, kwargs))
        if not config.log_model_signatures:
            _logger.info(
                "Signature is automatically generated for logged model if "
                "input_example is provided. To disable log_model_signatures, "
                "please also disable log_input_examples."
            )

    registered_model_name = get_autologging_config(
        mlflow.openai.FLAVOR_NAME, "registered_model_name", None
    )
    try:
        task = mlflow.openai._get_task_name(instance.__class__)
        with disable_autologging():
            # If the user is using `openai.OpenAI()` client,
            # they do not need to set the "OPENAI_API_KEY" environment variable.
            # This temporarily sets the API key as an environment variable
            # so that the model can be logged.
            with _set_api_key_env_var(instance._client):
                mlflow.openai.log_model(
                    kwargs.get("model"),
                    task,
                    "model",
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    run_id=run_id,
                )
    except Exception as e:
        _logger.warning(f"Failed to log model due to error: {e}")

    # Even if the model is not logged, we keep a single run per model
    instance._mlflow_model_logged = True


def _start_span(mlflow_client: MlflowClient, instance: Any, inputs: dict[str, Any], run_id: str):
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
        tm.set_request_metadata(span.request_id, TraceMetadataKey.SOURCE_RUN, run_id)

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
        mlflow_client.end_span(span.request_id, span.span_id, status=SpanStatusCode.ERROR)
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
        parsed = chunk.choices[0].delta.content or ""
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
