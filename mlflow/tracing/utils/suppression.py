import contextlib
from contextvars import ContextVar

from opentelemetry import context as otel_context

# True while MLflow exports traces. The span processor skips root spans started under it, which
# covers instrumentation that ignores OpenTelemetry's suppression key, such as storage SDKs'
# own tracing.
_IN_TRACE_EXPORT: ContextVar[bool] = ContextVar("mlflow_in_trace_export", default=False)


@contextlib.contextmanager
def suppress_export_tracing():
    """
    Keep MLflow's trace export from recording its own I/O as traces when it shares the tracer
    provider with other OpenTelemetry instrumentation.
    """
    token = _IN_TRACE_EXPORT.set(True)
    # `_SUPPRESS_INSTRUMENTATION_KEY` is private, but it is the key the OpenTelemetry SDK's own span
    # processors set around `export()`, and it exists in every opentelemetry-api MLflow supports.
    # The public `suppress_instrumentation()` lives in opentelemetry-instrumentation, which MLflow
    # does not depend on.
    otel_token = otel_context.attach(
        otel_context.set_value(otel_context._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        yield
    finally:
        otel_context.detach(otel_token)
        _IN_TRACE_EXPORT.reset(token)
