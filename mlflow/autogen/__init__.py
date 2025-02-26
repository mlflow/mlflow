import logging
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from mlflow.entities.span import create_mlflow_span
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import _get_tracer_provider
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import get_otel_attribute
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import _logger, autologging_integration, safe_patch

FLAVOR_NAME = "autogen"

_logger = logging.getLogger(__name__)


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    from autogen_core import SingleThreadedAgentRuntime

    tracer_provider = _get_tracer_provider()

    def patched_call(original, self, *args, **kwargs):
        if _tracer_provider := kwargs.get("tracer_provider"):
            _logger.warning(
                f"Tracer provider is provided to {original.__class__}."
                "The original provider is replaced by mlflow autologging and is not effective."
            )
            kwargs.pop("tracer_provider")
        return original(self, *args, **kwargs, tracer_provider=tracer_provider)

    safe_patch(FLAVOR_NAME, SingleThreadedAgentRuntime, "__init__", patched_call)

    try:
        from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost

        safe_patch(FLAVOR_NAME, GrpcWorkerAgentRuntimeHost, "__init__", patched_call)
    except ImportError:
        pass


class AutogenSpanProcessor(SimpleSpanProcessor):
    """
    An span processor that resister the span to the mlflow trace manager.
    This class fills the gap between Otel and mlflow tracing where we store the spans
    in memory until the root span is ended.
    """

    def __init__(self):
        pass

    def on_start(self, span, parent_context: Optional[Context] = None):
        request_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
        mlflow_span = create_mlflow_span(span, request_id)
        # TODO: convert native span attributes into mlflow attributes
        InMemoryTraceManager.get_instance().register_span(mlflow_span)

    def on_end(self, span: ReadableSpan) -> None:
        pass
