import logging

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import (
    NoOpTracerProvider,
    ProxyTracerProvider,
    get_tracer_provider,
    set_tracer_provider,
)

from mlflow.entities.span import create_mlflow_span
from mlflow.semantic_kernel.tracing_utils import set_span_type, set_token_usage
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import _get_tracer
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    _bypass_attribute_guard,
    get_mlflow_span_for_otel_span,
    get_otel_attribute,
)

_logger = logging.getLogger(__name__)


def _enable_experimental_genai_tracing():
    # NB: These settings are required to enable the telemetry for genai attributes
    # such as chat completion inputs/outputs, which are currently marked as experimental.
    # We directly update the singleton setting object instead of using env vars,
    # because the object might be already initialized by the time we call this function.
    # https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/telemetry-with-console
    from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
        MODEL_DIAGNOSTICS_SETTINGS,
    )

    MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics = True
    MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive = True

    try:
        # This only exists in Semantic Kernel 1.35.1 or later.
        from semantic_kernel.utils.telemetry.agent_diagnostics.decorators import (
            MODEL_DIAGNOSTICS_SETTINGS as AGENT_DIAGNOSTICS_SETTINGS,
        )

        AGENT_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics = True
        AGENT_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive = True
    except ImportError:
        pass

    _logger.info("Semantic Kernel Otel diagnostics is turned on for enabling tracing.")


def setup_semantic_kernel_tracing():
    _enable_experimental_genai_tracing()

    # NB: This logic has a known issue that it does not work when Semantic Kernel program is
    # executed before calling this setup is called. This is because Semantic Kernel caches the
    # tracer instance in each module (ref:https://github.com/microsoft/semantic-kernel/blob/6ecf2b9c2c893dc6da97abeb5962dfc49bed062d/python/semantic_kernel/functions/kernel_function.py#L46),
    # which prevent us from updating the span processor setup for the tracer.
    # Therefore, `mlflow.semantic_kernel.autolog()` should always be called before running the
    # Semantic Kernel program.

    provider = get_tracer_provider()
    sk_processor = SemanticKernelSpanProcessor()
    if isinstance(provider, (NoOpTracerProvider, ProxyTracerProvider)):
        new_provider = SDKTracerProvider()
        new_provider.add_span_processor(sk_processor)
        set_tracer_provider(new_provider)
    else:
        if not any(
            isinstance(p, SemanticKernelSpanProcessor)
            for p in provider._active_span_processor._span_processors
        ):
            provider.add_span_processor(sk_processor)


class SemanticKernelSpanProcessor(SimpleSpanProcessor):
    def __init__(self):
        # NB: Dummy NoOp exporter, because OTel span processor requires an exporter
        self.span_exporter = SpanExporter()

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        # Trigger MLflow's span processor
        tracer = _get_tracer(__name__)
        tracer.span_processor.on_start(span, parent_context)

        trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
        mlflow_span = create_mlflow_span(span, trace_id)

        # Register new span in the in-memory trace manager
        InMemoryTraceManager.get_instance().register_span(mlflow_span)

    def on_end(self, span: OTelReadableSpan) -> None:
        mlflow_span = get_mlflow_span_for_otel_span(span)
        if mlflow_span is None:
            _logger.debug("Span not found in the map. Skipping end.")
            return

        with _bypass_attribute_guard(mlflow_span._span):
            set_span_type(mlflow_span)
            set_token_usage(mlflow_span)

        # Export the span using MLflow's span processor
        tracer = _get_tracer(__name__)
        tracer.span_processor.on_end(span)
