import json
import logging

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.tracing.constant import (
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    SpanAttributeKey,
    TraceMetadataKey,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    _try_get_prediction_context,
    aggregate_usage_from_spans,
    deduplicate_span_names_in_place,
    generate_trace_id_v3,
    get_otel_attribute,
    maybe_get_dependencies_schemas,
    maybe_get_request_id,
    update_trace_state_from_span_conditionally,
)
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_MODEL_SERVING_ENDPOINT_NAME

_logger = logging.getLogger(__name__)


_HEADER_REQUEST_ID_KEY = "X-Request-Id"


# Extracting for testing purposes
def _get_flask_request():
    import flask

    if flask.has_request_context():
        return flask.request


class InferenceTableSpanProcessor(SimpleSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This processor is used when the tracing destination is Databricks Inference Table.
    """

    def __init__(self, span_exporter: SpanExporter):
        self.span_exporter = span_exporter
        self._trace_manager = InMemoryTraceManager.get_instance()

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        """
        Handle the start of a span. This method is called when an OpenTelemetry span is started.

        Args:
            span: An OpenTelemetry Span object that is started.
            parent_context: The context of the span. Note that this is only passed when the context
                object is explicitly specified to OpenTelemetry start_span call. If the parent
                span is obtained from the global context, it won't be passed here so we should not
                rely on it.
        """
        databricks_request_id = maybe_get_request_id()
        if databricks_request_id is None:
            # NB: This is currently used for streaming inference in Databricks Model Serving.
            # In normal prediction, serving logic pass the request ID using the
            # `with set_prediction_context` context manager that wraps `model.predict`
            # call. However, in streaming case, the context manager is not applicable
            # so we still need to rely on Flask request context (which is set to the
            # stream response via flask.stream_with_context()
            if flask_request := _get_flask_request():
                databricks_request_id = flask_request.headers.get(_HEADER_REQUEST_ID_KEY)
                if not databricks_request_id:
                    _logger.warning(
                        "Databricks request ID not found in the request headers. "
                        "Skipping trace processing."
                    )
                    return
            else:
                _logger.warning(
                    "Failed to get Databricks request ID from the request headers because "
                    "request context is not available. Skipping trace processing."
                )
                return

        trace_id = generate_trace_id_v3(span)
        span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(trace_id))
        tags = {}
        if dependencies_schema := maybe_get_dependencies_schemas():
            tags.update(dependencies_schema)

        if span._parent is None:
            trace_info = TraceInfo(
                trace_id=trace_id,
                client_request_id=databricks_request_id,
                # NB: Agent framework populate the MLFLOW_EXPERIMENT_ID environment variable
                #   with the experiment ID to which the model is logged. We don't use the
                #   _get_experiment_id() method because it will fallback to the default
                #   experiment if the MLFLOW_EXPERIMENT_ID is not set.
                trace_location=TraceLocation.from_experiment_id(MLFLOW_EXPERIMENT_ID.get()),
                request_time=span.start_time // 1_000_000,  # nanosecond to millisecond
                execution_duration=None,
                state=TraceState.IN_PROGRESS,
                trace_metadata=self._get_trace_metadata(),
                tags=tags,
            )
            self._trace_manager.register_trace(span.context.trace_id, trace_info)

    def on_end(self, span: OTelReadableSpan) -> None:
        """
        Handle the end of a span. This method is called when an OpenTelemetry span is ended.

        Args:
            span: An OpenTelemetry ReadableSpan object that is ended.
        """
        # Processing the trace only when the root span is found.
        if span._parent is not None:
            return

        trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
        with self._trace_manager.get_trace(trace_id) as trace:
            if trace is None:
                _logger.debug(f"Trace data with trace ID {trace_id} not found.")
                return

            trace.info.execution_duration = (span.end_time - span.start_time) // 1_000_000

            # Update trace state from span status, but only if the user hasn't explicitly set
            # a different trace status
            update_trace_state_from_span_conditionally(trace, span)

            spans = list(trace.span_dict.values())
            deduplicate_span_names_in_place(spans)

            # Aggregate token usage information from all spans
            if usage := aggregate_usage_from_spans(spans):
                trace.info.request_metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(usage)

        super().on_end(span)

    def _get_trace_metadata(self) -> dict[str, str]:
        metadata = {TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)}

        context = _try_get_prediction_context()
        if context:
            metadata[MLFLOW_DATABRICKS_MODEL_SERVING_ENDPOINT_NAME] = context.endpoint_name or ""

        # The model ID fetch order is
        # 1. from the context, this could be set by databricks serving
        # 2. from the active model, this could be set by model loading or with environment variable
        if context and context.model_id:
            metadata[TraceMetadataKey.MODEL_ID] = context.model_id
            _logger.debug(f"Model id fetched from the context: {context.model_id}")
        else:
            try:
                from mlflow.tracking.fluent import _get_active_model_id_global

                if active_model_id := _get_active_model_id_global():
                    metadata[TraceMetadataKey.MODEL_ID] = active_model_id
                    _logger.debug(
                        f"Adding active model ID {active_model_id} to the trace metadata."
                    )
            except Exception as e:
                _logger.debug(
                    f"Failed to get model ID from the active model: {e}. "
                    "Skipping adding model ID to trace metadata.",
                    exc_info=True,
                )
        return metadata
