"""Tests for OtelSpanProcessor metrics export functionality."""

import time

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import mlflow
from mlflow.tracing.processor.otel import OtelSpanProcessor


@pytest.fixture
def metric_reader():
    """Create an in-memory metric reader for testing."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    yield reader
    provider.shutdown()


def test_metrics_export(monkeypatch, metric_reader):
    """Test that metrics are exported with correct attributes."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090")

    OtelSpanProcessor(span_exporter=None, export_metrics=True)
    mlflow.set_experiment("test_experiment")

    @mlflow.trace(span_type="CHAIN", name="parent")
    def parent_function():
        mlflow.update_current_trace({"env": "test", "version": "1.0"})
        time.sleep(0.01)
        return child_function()

    @mlflow.trace(span_type="LLM", name="child")
    def child_function():
        time.sleep(0.01)
        return "result"

    @mlflow.trace(span_type="TOOL", name="error_function")
    def error_function():
        time.sleep(0.01)
        raise ValueError("Test error")

    # Execute successful trace
    assert parent_function() == "result"

    # Execute error trace
    with pytest.raises(ValueError, match="Test error"):
        error_function()

    # Verify metrics
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None

    found_metrics = False
    span_types_seen = set()
    statuses_seen = set()

    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "mlflow.trace.span.duration":
                    found_metrics = True
                    assert metric.unit == "ms"
                    assert len(metric.data.data_points) > 0

                    for dp in metric.data.data_points:
                        attrs = dict(dp.attributes)
                        span_types_seen.add(attrs["span_type"])
                        statuses_seen.add(attrs["span_status"])
                        assert dp.sum >= 10  # At least 10ms duration

                        # Check tags on root span
                        if attrs["root"] == "True" and attrs["span_type"] == "CHAIN":
                            assert attrs.get("tags.env") == "test"
                            assert attrs.get("tags.version") == "1.0"

    assert found_metrics, "No metrics found"
    assert span_types_seen == {"CHAIN", "LLM", "TOOL"}
    assert "OK" in statuses_seen
    assert "ERROR" in statuses_seen


def test_no_metrics_when_disabled(monkeypatch, metric_reader):
    """Test that no metrics are collected when disabled."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)

    OtelSpanProcessor(span_exporter=None, export_metrics=False)

    @mlflow.trace(name="test")
    def test_function():
        return "result"

    test_function()

    metrics_data = metric_reader.get_metrics_data()
    if metrics_data:
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    assert metric.name != "mlflow.trace.span.duration"
