from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter


class OtelSpanProcessor(BatchSpanProcessor):
    """
    SpanProcessor implementation to export MLflow traces to a OpenTelemetry collector.

    Extending OpenTelemetry BatchSpanProcessor to add some custom hooks to be executed when a span
    is started or ended (before exporting).
    """

    def __init__(self, span_exporter: SpanExporter):
        super().__init__(span_exporter)
        # In opentelemetry-sdk 1.34.0, the `span_exporter` field was removed from the
        # `BatchSpanProcessor` class.
        # https://github.com/open-telemetry/opentelemetry-python/issues/4616
        #
        # The `span_exporter` field was restored as a property in 1.34.1
        # https://github.com/open-telemetry/opentelemetry-python/pull/4621
        #
        # We use a try-except block to maintain compatibility with both versions.
        try:
            self.span_exporter = span_exporter
        except AttributeError:
            pass
