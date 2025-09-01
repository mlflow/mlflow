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
        time.sleep(0.01)  # 10ms
        return child_function()

    @mlflow.trace(span_type="LLM", name="child")
    def child_function():
        time.sleep(0.25)  # 250ms
        return "result"

    @mlflow.trace(span_type="TOOL", name="error_function")
    def error_function():
        time.sleep(1.0)  # 1000ms
        raise ValueError("Test error")

    # Execute successful trace
    assert parent_function() == "result"

    # Execute error trace
    with pytest.raises(ValueError, match="Test error"):
        error_function()

    # Verify metrics
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None

    # Collect all data points
    data_points = []
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "mlflow.trace.span.duration":
                    assert metric.unit == "ms"
                    data_points.extend(metric.data.data_points)

    assert len(data_points) == 3, "Expected exactly 3 span metrics"

    # Sort by span type for predictable ordering
    data_points.sort(key=lambda dp: dict(dp.attributes)["span_type"])

    # Check each metric
    chain_metric, llm_metric, tool_metric = data_points

    # CHAIN span (parent) - includes child time, so ~260ms total
    chain_metric_attrs = dict(chain_metric.attributes)
    assert chain_metric_attrs["span_type"] == "CHAIN"
    assert chain_metric_attrs["span_status"] == "OK"
    assert chain_metric_attrs["root"] == "True"
    assert chain_metric_attrs["tags.env"] == "test"
    assert chain_metric_attrs["tags.version"] == "1.0"
    assert chain_metric.sum >= 260

    # LLM span (child) - 250ms
    llm_metric_attrs = dict(llm_metric.attributes)
    assert llm_metric_attrs["span_type"] == "LLM"
    assert llm_metric_attrs["span_status"] == "OK"
    assert llm_metric_attrs["root"] == "False"
    assert llm_metric.sum >= 250

    # TOOL span (error) - 1000ms
    tool_metric_attrs = dict(tool_metric.attributes)
    assert tool_metric_attrs["span_type"] == "TOOL"
    assert tool_metric_attrs["span_status"] == "ERROR"
    assert tool_metric_attrs["root"] == "True"
    assert tool_metric.sum >= 1000


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
