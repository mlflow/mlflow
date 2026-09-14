"""Prometheus metrics for the OTLP trace ingestion endpoint.

Metrics are registered on the default ``prometheus_client`` global registry,
which is the same registry used by the Flask-based exporter
(``prometheus_flask_exporter``).  They therefore appear automatically on the
existing ``/metrics`` endpoint when ``--expose-prometheus`` is enabled.

If ``prometheus_client`` is not installed (it is an optional dependency pulled
in by ``prometheus-flask-exporter``), every metric object falls back to
``None`` and callers must guard usage with ``if METRIC is not None``.
"""

try:
    from prometheus_client import Counter, Histogram

    SPANS_INGESTED = Counter(
        "mlflow_spans_ingested_total",
        "Total number of spans successfully ingested",
        ["protocol"],
    )

    INGESTION_ERRORS = Counter(
        "mlflow_trace_ingestion_server_errors_total",
        "Total number of server errors during trace ingestion",
        ["protocol"],
    )

    REQUEST_DURATION = Histogram(
        "mlflow_otlp_request_duration_seconds",
        "End-to-end duration of OTLP export_traces requests",
    )

    REQUEST_PAYLOAD_BYTES = Histogram(
        "mlflow_otlp_request_payload_bytes",
        "Size of raw OTLP request body (before decompression)",
        buckets=(
            1_024,
            5_120,
            20_480,
            102_400,
            524_288,
            1_048_576,
            5_242_880,
            10_485_760,
            33_554_432,
            float("inf"),
        ),
    )

    SPANS_PER_REQUEST = Histogram(
        "mlflow_otlp_spans_per_request",
        "Number of spans in each OTLP export request",
        buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, float("inf")),
    )

except ImportError:
    SPANS_INGESTED = None
    INGESTION_ERRORS = None
    REQUEST_DURATION = None
    REQUEST_PAYLOAD_BYTES = None
    SPANS_PER_REQUEST = None
