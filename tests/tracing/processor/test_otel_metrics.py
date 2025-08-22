"""Tests for OtelSpanProcessor metrics export functionality."""

import os
from unittest import mock

import pytest

import mlflow
from mlflow.tracing.processor.otel import OtelSpanProcessor

try:
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.trace import StatusCode
except ImportError:
    pytest.skip("OTLP metric exporters are not installed", allow_module_level=True)


def create_mock_span(
    trace_id="test-trace-id",
    start_time=1000000000,
    end_time=1100000000,
    span_type="LLM",
    status=StatusCode.OK,
    is_root=True,
):
    """Create a mock span with common test properties."""
    mock_span = mock.MagicMock()
    mock_span.parent = None if is_root else mock.MagicMock()
    mock_span.context.trace_id = trace_id
    mock_span.start_time = start_time
    mock_span.end_time = end_time
    mock_span.status.status_code = status
    mock_span.attributes = {"mlflow.spanType": f'"{span_type}"'}
    return mock_span


@pytest.fixture
def mock_metric_reader():
    """Create an in-memory metric reader for testing."""
    return InMemoryMetricReader()


@pytest.fixture
def otel_metrics_env(monkeypatch, mock_metric_reader):
    """Set up environment for OTEL metrics export testing."""
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090/api/v1/otlp/v1/metrics"
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "http/protobuf")

    yield
    if "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT" in os.environ:
        del os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]
    if "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL" in os.environ:
        del os.environ["OTEL_EXPORTER_OTLP_METRICS_PROTOCOL"]


def test_otel_metrics_processor_initialization(otel_metrics_env):
    """Test that OtelSpanProcessor can be initialized with metrics export enabled."""
    mock_histogram = mock.MagicMock()

    with mock.patch(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter"
    ) as mock_exporter:
        mock_exporter.return_value = mock.MagicMock()

        with mock.patch("opentelemetry.sdk.metrics.MeterProvider"):
            with mock.patch("opentelemetry.metrics.get_meter") as mock_get_meter:
                mock_meter = mock.MagicMock()
                mock_meter.create_histogram.return_value = mock_histogram
                mock_get_meter.return_value = mock_meter

                OtelSpanProcessor(
                    span_exporter=mock.MagicMock(),
                    export_spans=False,
                    export_metrics=True,
                )

                mock_exporter.assert_called_once_with(
                    endpoint="http://localhost:9090/api/v1/otlp/v1/metrics"
                )

                mock_meter.create_histogram.assert_called_once_with(
                    name="mlflow.trace.span.duration",
                    description="Duration of spans in milliseconds",
                    unit="ms",
                )


def test_metrics_collection_from_spans(otel_metrics_env):
    """Test that metrics are collected from spans when metrics export is enabled."""
    mock_histogram = mock.MagicMock()

    with mock.patch(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter"
    ) as mock_exporter:
        mock_exporter.return_value = mock.MagicMock()

        with mock.patch("opentelemetry.sdk.metrics.MeterProvider"):
            with mock.patch("opentelemetry.metrics.get_meter") as mock_get_meter:
                mock_meter = mock.MagicMock()
                mock_meter.create_histogram.return_value = mock_histogram
                mock_get_meter.return_value = mock_meter

                OtelSpanProcessor(
                    span_exporter=mock.MagicMock(), export_spans=False, export_metrics=True
                )

                mlflow.set_experiment("test_metrics")

                @mlflow.trace(span_type="LLM", name="test_llm_call")
                def test_function():
                    span = mlflow.get_current_active_span()
                    if span:
                        span.set_attributes(
                            {
                                "model_name": "gpt-4",
                                "input_tokens": 50,
                                "output_tokens": 25,
                                "total_tokens": 75,
                            }
                        )
                    return "test response"

                result = test_function()
                assert result == "test response"
                assert mock_histogram.record.called

                call_args = mock_histogram.record.call_args
                duration_ms = call_args[0][0]
                attributes = call_args[1]["attributes"]

                assert duration_ms > 0
                assert attributes["span_type"] == "LLM"
                assert attributes["span_status"] == "OK"
                assert attributes["root"] == "True"
                assert "experiment_id" in attributes


def test_metrics_only_no_trace_cleanup(otel_metrics_env):
    """Test that when only exporting metrics, traces are not cleaned up inappropriately."""
    mock_trace_manager = mock.MagicMock()
    mock_histogram = mock.MagicMock()

    with mock.patch(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter"
    ) as mock_exporter:
        mock_exporter.return_value = mock.MagicMock()

        with mock.patch("opentelemetry.sdk.metrics.MeterProvider"):
            with mock.patch("opentelemetry.metrics.get_meter") as mock_get_meter:
                mock_meter = mock.MagicMock()
                mock_meter.create_histogram.return_value = mock_histogram
                mock_get_meter.return_value = mock_meter

                with mock.patch(
                    "mlflow.tracing.processor.otel.InMemoryTraceManager.get_instance"
                ) as mock_get_instance:
                    mock_get_instance.return_value = mock_trace_manager

                    processor = OtelSpanProcessor(
                        span_exporter=mock.MagicMock(), export_spans=False, export_metrics=True
                    )

                    mock_span = create_mock_span(trace_id="test-trace-id")
                    processor.on_end(mock_span)

                    mock_trace_manager.pop_trace.assert_not_called()
                    mock_trace_manager.get_mlflow_trace_id_from_otel_id.assert_called_with(
                        "test-trace-id"
                    )
                    mock_histogram.record.assert_called_once()


def test_span_type_json_decoding(otel_metrics_env):
    """Test that JSON-encoded span types are properly decoded for metrics."""
    mock_histogram = mock.MagicMock()

    with mock.patch(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter"
    ) as mock_exporter:
        mock_exporter.return_value = mock.MagicMock()

        with mock.patch("opentelemetry.sdk.metrics.MeterProvider"):
            with mock.patch("opentelemetry.metrics.get_meter") as mock_get_meter:
                mock_meter = mock.MagicMock()
                mock_meter.create_histogram.return_value = mock_histogram
                mock_get_meter.return_value = mock_meter

                processor = OtelSpanProcessor(
                    span_exporter=mock.MagicMock(), export_spans=False, export_metrics=True
                )

                mock_span = create_mock_span(span_type="LLM")
                processor.on_end(mock_span)

                assert mock_histogram.record.called
                call_args = mock_histogram.record.call_args
                attributes = call_args[1]["attributes"]
                assert attributes["span_type"] == "LLM"


def test_metrics_with_comprehensive_attributes(otel_metrics_env):
    """Test that metrics include comprehensive trace context attributes."""
    mock_histogram = mock.MagicMock()
    mock_trace_manager = mock.MagicMock()

    mock_trace = mock.MagicMock()
    mock_trace.info.tags = {
        "environment": "production",
        "user_tier": "premium",
        "region": "us-west-2",
    }
    mock_trace.info.trace_metadata = {
        "user_id": "user123",
        "session_id": "session456",
        "app_version": "v2.1.0",
    }

    mock_trace_manager.get_mlflow_trace_id_from_otel_id.return_value = "mlflow-trace-123"
    mock_trace_manager.get_trace.return_value.__enter__.return_value = mock_trace

    with mock.patch(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter"
    ) as mock_exporter:
        mock_exporter.return_value = mock.MagicMock()

        with mock.patch("opentelemetry.sdk.metrics.MeterProvider"):
            with mock.patch("opentelemetry.metrics.get_meter") as mock_get_meter:
                mock_meter = mock.MagicMock()
                mock_meter.create_histogram.return_value = mock_histogram
                mock_get_meter.return_value = mock_meter

                with mock.patch(
                    "mlflow.tracing.processor.otel.InMemoryTraceManager.get_instance"
                ) as mock_get_instance:
                    mock_get_instance.return_value = mock_trace_manager

                    processor = OtelSpanProcessor(
                        span_exporter=mock.MagicMock(), export_spans=False, export_metrics=True
                    )

                    mock_span = create_mock_span(
                        trace_id="otel-trace-id",
                        start_time=1000000000,
                        end_time=1200000000,
                        span_type="CHAIN",
                    )

                    processor.on_end(mock_span)

                    assert mock_histogram.record.called
                    call_args = mock_histogram.record.call_args
                    duration_ms = call_args[0][0]
                    attributes = call_args[1]["attributes"]

                    assert duration_ms == 200.0
                    assert attributes["span_type"] == "CHAIN"
                    assert attributes["span_status"] == "OK"
                    assert attributes["root"] == "True"
                    assert attributes["tags.environment"] == "production"
                    assert attributes["tags.user_tier"] == "premium"
                    assert attributes["tags.region"] == "us-west-2"
                    assert attributes["metadata.user_id"] == "user123"
                    assert attributes["metadata.session_id"] == "session456"
                    assert attributes["metadata.app_version"] == "v2.1.0"


def test_no_metrics_when_disabled(otel_metrics_env):
    """Test that no metrics are collected when metrics export is disabled."""
    mock_histogram = mock.MagicMock()

    with mock.patch("opentelemetry.sdk.metrics.MeterProvider"):
        with mock.patch("opentelemetry.metrics.get_meter") as mock_get_meter:
            mock_meter = mock.MagicMock()
            mock_meter.create_histogram.return_value = mock_histogram
            mock_get_meter.return_value = mock_meter

            processor = OtelSpanProcessor(
                span_exporter=mock.MagicMock(),
                export_spans=True,
                export_metrics=False,
            )

            mock_span = create_mock_span()
            processor.on_end(mock_span)
            mock_histogram.record.assert_not_called()
