import logging
from typing import Optional

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

from mlflow.entities import SpanType
from mlflow.semantic_kernel.tracing_utils import (
    _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN,
    _get_span_type,
    _set_token_usage,
)
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context

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

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
        otel_span_id = span.get_span_context().span_id
        parent_span_id = span.parent.span_id if span.parent else None
        parent_st = _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN.get(parent_span_id)
        parent_span = parent_st[0] if parent_st else None

        mlflow_span = start_span_no_context(
            name=span.name,
            parent_span=parent_span,
            attributes=dict(span.attributes) if span.attributes else None,
        )
        token = set_span_in_context(mlflow_span)
        _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN[otel_span_id] = (mlflow_span, token)

    def on_end(self, span: OTelReadableSpan) -> None:
        st = _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN.pop(span.get_span_context().span_id, None)
        if st is None:
            _logger.debug("Span not found in the map. Skipping end.")
            return

        mlflow_span, token = st
        mlflow_span.set_attributes(dict(span.attributes))
        _set_token_usage(mlflow_span, span.attributes)

        if not mlflow_span.span_type or mlflow_span.span_type == SpanType.UNKNOWN:
            mlflow_span.set_span_type(_get_span_type(span))

        detach_span_from_context(token)
        mlflow_span.end()
