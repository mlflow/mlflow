import logging
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import mlflow
from mlflow.entities.span import create_mlflow_span
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import _get_tracer_provider
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import get_otel_attribute
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    _logger,
    autologging_integration,
    get_autologging_config,
    safe_patch,
)

FLAVOR_NAME = "autogen"

_logger = logging.getLogger(__name__)


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
    from autogen_core import SingleThreadedAgentRuntime

    tracer_provider = _get_tracer_provider()
    tracer_provider.add_span_processor(AutogenSpanProcessor())

    def patched_init(original, self, *args, **kwargs):
        if not get_autologging_config(FLAVOR_NAME, "log_traces"):
            return original(self, *args, **kwargs)

        if _tracer_provider := kwargs.get("tracer_provider"):
            _logger.warning(
                f"Tracer provider is provided to {original.__class__}."
                "The original provider has been replaced by MLflow autologging."
            )
            kwargs.pop("tracer_provider")
        return original(self, *args, **kwargs, tracer_provider=tracer_provider)

    async def patched_run(original, self, *args, **kwargs):
        if not get_autologging_config(FLAVOR_NAME, "log_traces"):
            return await original(self, *args, **kwargs)
        else:
            return await mlflow.trace(original)(self, *args, **kwargs)

    safe_patch(FLAVOR_NAME, SingleThreadedAgentRuntime, "__init__", patched_init)
    safe_patch(FLAVOR_NAME, AssistantAgent, "run", patched_run)
    safe_patch(FLAVOR_NAME, AssistantAgent, "on_messages", patched_run)

    try:
        from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntimeHost

        safe_patch(FLAVOR_NAME, GrpcWorkerAgentRuntimeHost, "__init__", patched_init)
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

    def on_start(self, span: Span, parent_context: Optional[Context] = None):
        request_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
        mlflow_span = create_mlflow_span(span, request_id)
        # Set span attributes logged by Autogen to the MLflow span.
        # We shouldn't overwrite the attributes with `mlflow.` prefix as they are treated separately.
        mlflow_span.set_attributes(
            {
                key: value
                for key, value in dict(span.attributes).items()
                if not key.startswith("mlflow.")
            }
        )
        # TODO: operation type publish and create does not contain useful information,
        # consider how to exlucde noisy spans.
        
        # `message` attribute is used for the input message to handlers
        if message := mlflow_span.attributes.get("message"):
            mlflow_span.set_inputs(message)
        InMemoryTraceManager.get_instance().register_span(mlflow_span)

    def on_end(self, span: ReadableSpan) -> None:
        # traces are exported through BaseMlflowSpanProcessor.on_end
        pass
