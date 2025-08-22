"""Tests for OtelSpanProcessor metrics export functionality."""

import os
import time
from unittest import mock

import pytest

import mlflow
from mlflow.tracing.processor.otel import OtelSpanProcessor

# OTLP exporters are not installed in some CI jobs
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
    mock_span.attributes = {"mlflow.spanType": f'"{span_type}"'}  # JSON-encoded
    return mock_span


@pytest.fixture
def mock_metric_reader():
    """Create an in-memory metric reader for testing."""
    return InMemoryMetricReader()


@pytest.fixture
def otel_metrics_env(monkeypatch, mock_metric_reader):
    """Set up environment for OTEL metrics export testing."""
    # Set up fake OTLP metrics endpoint
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090/api/v1/otlp/v1/metrics"
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "http/protobuf")

    yield

    # Clean up
    if "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT" in os.environ:
        del os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]
    if "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL" in os.environ:
        del os.environ["OTEL_EXPORTER_OTLP_METRICS_PROTOCOL"]


def test_otel_metrics_processor_initialization(otel_metrics_env):
    """Test that OtelSpanProcessor can be initialized with metrics export enabled."""
    mock_histogram = mock.MagicMock()

    # Mock the metric exporter module import to avoid network calls
    with mock.patch(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter"
    ) as mock_exporter:
        mock_exporter.return_value = mock.MagicMock()

        with mock.patch("opentelemetry.sdk.metrics.MeterProvider"):
            with mock.patch("opentelemetry.metrics.get_meter") as mock_get_meter:
                mock_meter = mock.MagicMock()
                mock_meter.create_histogram.return_value = mock_histogram
                mock_get_meter.return_value = mock_meter

                # Create processor with metrics export enabled
                OtelSpanProcessor(
                    span_exporter=mock.MagicMock(),
                    export_spans=False,  # Only export metrics
                    export_metrics=True,
                )

                # Verify OTLP metric exporter was created (behavior test)
                mock_exporter.assert_called_once_with(
                    endpoint="http://localhost:9090/api/v1/otlp/v1/metrics"
                )

                # Verify histogram was created (behavior test)
                mock_meter.create_histogram.assert_called_once_with(
                    name="mlflow.trace.span.duration",
                    description="Duration of spans in milliseconds",
                    unit="ms",
                )


def test_metrics_collection_from_spans(otel_metrics_env):
    """Test that metrics are collected from spans when metrics export is enabled."""

    # Create a mock histogram to capture metrics
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

                # Create processor with only metrics export
                OtelSpanProcessor(
                    span_exporter=mock.MagicMock(), export_spans=False, export_metrics=True
                )

                # Set up MLflow experiment
                mlflow.set_experiment("test_metrics")

                @mlflow.trace(span_type="LLM", name="test_llm_call")
                def test_function():
                    time.sleep(0.1)  # 100ms span
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

                # Execute the traced function
                result = test_function()
                assert result == "test response"

                # Wait a moment for span processing
                time.sleep(0.2)

                # Verify histogram was called to record metrics
                assert mock_histogram.record.called

                # Check the recorded metrics call
                call_args = mock_histogram.record.call_args
                duration_ms = call_args[0][0]  # First positional argument
                attributes = call_args[1]["attributes"]  # Keyword argument

                # Verify duration is a positive number (don't assert on exact timing)
                assert duration_ms > 0

                # Verify attributes include span metadata
                assert attributes["span_type"] == "LLM"
                assert attributes["span_status"] == "OK"
                assert attributes["root"] == "True"
                assert "experiment_id" in attributes

                # Verify span attributes are included (if available)
                # Note: This depends on trace registration working properly


def test_metrics_only_no_trace_cleanup(otel_metrics_env):
    """Test that when only exporting metrics, traces are not cleaned up inappropriately."""

    # Mock trace manager to verify trace cleanup behavior
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

                    # Create processor with only metrics export (no span export)
                    processor = OtelSpanProcessor(
                        span_exporter=mock.MagicMock(), export_spans=False, export_metrics=True
                    )

                    # Create a mock span using the utility
                    mock_span = create_mock_span(trace_id="test-trace-id")

                    # Call on_end to trigger metrics collection
                    processor.on_end(mock_span)

                    # Verify that pop_trace was NOT called (behavior test for the critical bug fix)
                    mock_trace_manager.pop_trace.assert_not_called()

                    # Verify that trace registration mapping was used for metrics context
                    mock_trace_manager.get_mlflow_trace_id_from_otel_id.assert_called_with(
                        "test-trace-id"
                    )

                    # Verify metrics were recorded
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

                # Create a mock span using the utility
                mock_span = create_mock_span(span_type="LLM")

                processor.on_end(mock_span)

                # Verify histogram was called
                assert mock_histogram.record.called

                # Check that span type was decoded from JSON
                call_args = mock_histogram.record.call_args
                attributes = call_args[1]["attributes"]
                assert attributes["span_type"] == "LLM"  # Should be decoded, not '"LLM"'


def test_metrics_with_comprehensive_attributes(otel_metrics_env):
    """Test that metrics include comprehensive trace context attributes."""

    mock_histogram = mock.MagicMock()
    mock_trace_manager = mock.MagicMock()

    # Set up mock trace with tags and metadata
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

                    # Create a mock span using the utility (200ms duration)
                    mock_span = create_mock_span(
                        trace_id="otel-trace-id",
                        start_time=1000000000,
                        end_time=1200000000,
                        span_type="CHAIN",
                    )

                    processor.on_end(mock_span)

                    # Verify histogram was called with comprehensive attributes
                    assert mock_histogram.record.called
                    call_args = mock_histogram.record.call_args
                    duration_ms = call_args[0][0]  # First positional argument
                    attributes = call_args[1]["attributes"]  # Keyword argument

                    # Verify duration is calculated correctly (200ms from mock span times)
                    assert duration_ms == 200.0

                    # Check basic span attributes
                    assert attributes["span_type"] == "CHAIN"
                    assert attributes["span_status"] == "OK"
                    assert attributes["root"] == "True"

                    # Check trace tags (prefixed with 'tags.')
                    assert attributes["tags.environment"] == "production"
                    assert attributes["tags.user_tier"] == "premium"
                    assert attributes["tags.region"] == "us-west-2"

                    # Check trace metadata (prefixed with 'metadata.')
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

            # Create processor with metrics export disabled
            processor = OtelSpanProcessor(
                span_exporter=mock.MagicMock(),
                export_spans=True,  # Only export spans
                export_metrics=False,  # No metrics
            )

            # Create a mock span using the utility
            mock_span = create_mock_span()

            # Should not raise any errors when processing span
            processor.on_end(mock_span)  # Should work without metrics recording

            # Verify no metrics were recorded (behavior test)
            mock_histogram.record.assert_not_called()
