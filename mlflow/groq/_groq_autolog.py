import logging
from typing import Iterator

import mlflow
from mlflow.entities import RunTag, SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

_logger = logging.getLogger(__name__)


def _get_span_type(task) -> str:
    from groq.resources.chat.completions import Completions as ChatCompletions
    from groq.resources.embeddings import Embeddings

    span_type_mapping = {
        ChatCompletions: SpanType.CHAT_MODEL,
        Embeddings: SpanType.EMBEDDING,
    }
    return span_type_mapping.get(task, SpanType.UNKNOWN)


def patched_call(original, self, *args, **kwargs):
    from openai import Stream
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.completion import Completion

    config = AutoLoggingConfig.init(flavor_name=mlflow.groq.FLAVOR_NAME)
    active_run = mlflow.active_run()

    # Active run should always take precedence over the run_id stored in the model
    run_id = active_run.info.run_id if active_run else getattr(self, "_mlflow_run_id", None)

    mlflow_client = mlflow.MlflowClient()
    request_id = None

    # If optional artifacts logging are enabled e.g. log_models, we need to create a run
    if config.should_log_optional_artifacts():
        # include run context tags
        resolved_tags = context_registry.resolve_tags(config.extra_tags)
        tags = _resolve_extra_tags(mlflow.groq.FLAVOR_NAME, resolved_tags)
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

    if config.log_traces:
        # Record input parameters to attributes
        attributes = {k: v for k, v in kwargs.items() if k != "messages"}

        # If there is an active span, create a child span under it, otherwise create a new trace
        if active_span := mlflow.get_current_active_span():
            span = mlflow_client.start_span(
                name=self.__class__.__name__,
                request_id=active_span.request_id,
                parent_id=active_span.span_id,
                span_type=_get_span_type(self.__class__),
                inputs=kwargs,
                attributes=attributes,
            )
        else:
            span = mlflow_client.start_trace(
                name=self.__class__.__name__,
                span_type=_get_span_type(self.__class__),
                inputs=kwargs,
                attributes=attributes,
            )

        request_id = span.request_id
        # Associate run ID to the trace manually, because if a new run is created by
        # autologging, it is not set as the active run thus not automatically
        # associated with the trace.
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
                chunk_dicts = [chunk.to_dict() for chunk in chunks]
                if config.log_traces and request_id:
                    mlflow_client.end_trace(
                        request_id=request_id,
                        attributes={"events": chunk_dicts},
                        outputs="".join(output),
                    )
            except Exception as e:
                _logger.warning(f"Encountered unexpected error during groq autologging: {e}")

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

    # Even if the model is not logged, we keep a single run per model
    self._mlflow_run_id = run_id

    # Terminate the run if it is not managed by the user
    if run_id is not None and (active_run is None or active_run.info.run_id != run_id):
        mlflow_client.set_terminated(run_id)

    return result
