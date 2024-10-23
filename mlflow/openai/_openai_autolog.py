import functools
import json
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator

from packaging.version import Version

import mlflow
from mlflow import MlflowException
from mlflow.entities import RunTag, SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
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


class _OpenAIJsonEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def _get_span_type(task) -> str:
    from openai.resources.chat.completions import Completions as ChatCompletions
    from openai.resources.completions import Completions
    from openai.resources.embeddings import Embeddings

    span_type_mapping = {
        ChatCompletions: SpanType.CHAT_MODEL,
        Completions: SpanType.LLM,
        Embeddings: SpanType.EMBEDDING,
    }
    return span_type_mapping.get(task, SpanType.UNKNOWN)


def patched_call(original, self, *args, **kwargs):
    from openai import Stream
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.completion import Completion

    config = AutoLoggingConfig.init(flavor_name=mlflow.openai.FLAVOR_NAME)
    run_id = getattr(self, "_mlflow_run_id", None)
    active_run = mlflow.active_run()
    mlflow_client = mlflow.MlflowClient()
    request_id = None

    # If optional artifacts logging are enabled e.g. log_models, we need to create a run
    if config.should_log_optional_artifacts() and run_id is None:
        # include run context tags
        resolved_tags = context_registry.resolve_tags(config.extra_tags)
        tags = _resolve_extra_tags(mlflow.openai.FLAVOR_NAME, resolved_tags)
        if active_run:
            run_id = active_run.info.run_id
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

    if config.log_traces:
        # If there is an active span, create a child span under it, otherwise create a new trace
        if active_span := mlflow.get_current_active_span():
            span = mlflow_client.start_span(
                name=self.__class__.__name__,
                request_id=active_span.request_id,
                parent_id=active_span.span_id,
                span_type=_get_span_type(self.__class__),
                inputs=kwargs,
            )
        else:
            span = mlflow_client.start_trace(
                name=self.__class__.__name__,
                span_type=_get_span_type(self.__class__),
                inputs=kwargs,
            )

        request_id = span.request_id
        # If a new autolog run is created, associate the trace with the run
        if run_id is not None:
            tm = InMemoryTraceManager().get_instance()
            tm.set_request_metadata(request_id, TraceMetadataKey.SOURCE_RUN, run_id)

    # Execute the original function
    try:
        result = original(self, *args, **kwargs)
    except Exception as e:
        # We have to end the trace even the exception is raised
        if config.log_traces and request_id:
            try:
                span.add_event(SpanEvent.from_exception(e))
                mlflow_client.end_trace(request_id=request_id, status=SpanStatusCode.ERROR)
            except Exception as inner_e:
                _logger.warning(f"Encountered unexpected error when ending trace: {inner_e}")
        raise e

    if isinstance(result, Stream):
        # If the output is a stream, we add a hook to store the intermediate chunks
        # and then log the outputs as a single artifact when the stream ends
        def _stream_output_logging_hook(stream: Iterator) -> Iterator:
            chunks = []
            output = []
            for chunk in stream:
                # `chunk.choices` can be empty: https://github.com/mlflow/mlflow/issues/13361
                if isinstance(chunk, Completion) and chunk.choices:
                    output.append(chunk.choices[0].text or "")
                elif isinstance(chunk, ChatCompletionChunk) and chunk.choices:
                    output.append(chunk.choices[0].delta.content or "")
                chunks.append(chunk)
                yield chunk

            try:
                chunk_dicts = []
                chunk_dicts = [chunk.to_dict() for chunk in chunks]
                if config.log_traces and request_id:
                    mlflow_client.end_trace(
                        request_id=request_id,
                        attributes={"events": chunk_dicts},
                        outputs="".join(output),
                    )
            except Exception as e:
                _logger.warning(f"Encountered unexpected error during openai autologging: {e}")

        result._iterator = _stream_output_logging_hook(result._iterator)
    else:
        if config.log_traces and request_id:
            try:
                if span.parent_id is None:
                    mlflow_client.end_trace(request_id=request_id, outputs=result)
                else:
                    mlflow_client.end_span(
                        request_id=request_id, span_id=span.span_id, outputs=result
                    )
            except Exception as e:
                _logger.warning(f"Encountered unexpected error when ending trace: {e}")

    input_example = None
    if config.log_models and not hasattr(self, "_mlflow_model_logged"):
        if config.log_input_examples:
            input_example = deepcopy(_get_input_from_model(self, kwargs))
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
            task = mlflow.openai._get_task_name(self.__class__)
            with disable_autologging():
                # If the user is using `openai.OpenAI()` client,
                # they do not need to set the "OPENAI_API_KEY" environment variable.
                # This temporarily sets the API key as an environment variable
                # so that the model can be logged.
                with _set_api_key_env_var(self._client):
                    mlflow.openai.log_model(
                        kwargs.get("model", None),
                        task,
                        "model",
                        input_example=input_example,
                        registered_model_name=registered_model_name,
                        run_id=run_id,
                    )
        except Exception as e:
            _logger.warning(f"Failed to log model due to error: {e}")
        self._mlflow_model_logged = True

    # Even if the model is not logged, we keep a single run per model
    if not hasattr(self, "_mlflow_run_id"):
        self._mlflow_run_id = run_id

    # Terminate the run if it is not managed by the user
    if run_id is not None and (active_run is None or active_run.info.run_id != run_id):
        mlflow_client.set_terminated(run_id)

    return result


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
