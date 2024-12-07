import logging

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

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
    config = AutoLoggingConfig.init(flavor_name=mlflow.groq.FLAVOR_NAME)
    active_run = mlflow.active_run()

    # Active run should always take precedence over the run_id stored in the model
    run_id = active_run.info.run_id if active_run else getattr(self, "_mlflow_run_id", None)

    mlflow_client = mlflow.MlflowClient()
    request_id = None

    if config.log_traces:
        # Record input parameters to attributes
        attributes = {k: v for k, v in kwargs.items() if k not in ["messages", "file"]}

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

    if config.log_traces and request_id:
        try:
            if span.parent_id is None:
                mlflow_client.end_trace(request_id=request_id, outputs=result)
            else:
                mlflow_client.end_span(request_id=request_id, span_id=span.span_id, outputs=result)
        except Exception as e:
            _logger.warning(f"Encountered unexpected error when ending trace: {e}")

    # Even if the model is not logged, we keep a single run per model
    self._mlflow_run_id = run_id

    # Terminate the run if it is not managed by the user
    if run_id is not None and (active_run is None or active_run.info.run_id != run_id):
        mlflow_client.set_terminated(run_id)

    return result
