import asyncio
import json

from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags

import mlflow
from mlflow.entities.span import SpanStatus, SpanStatusCode, create_mlflow_span
from mlflow.tracing.client import TracingClient

from tests.tracing.helper import skip_when_testing_trace_sdk


@skip_when_testing_trace_sdk
def test_tracing_client_link_prompt_versions_to_trace():
    with mlflow.start_run():
        # Register a prompt
        prompt_version = mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

        # Create a trace
        with mlflow.start_span("test_span"):
            trace_id = mlflow.get_active_trace_id()

        # Link prompts to trace
        client = TracingClient()
        client.link_prompt_versions_to_trace(trace_id, [prompt_version])

        # Verify the linked prompts tag was set
        trace = mlflow.get_trace(trace_id)
        assert "mlflow.linkedPrompts" in trace.info.tags

        # Parse and verify the linked prompts
        linked_prompts = json.loads(trace.info.tags["mlflow.linkedPrompts"])
        assert len(linked_prompts) == 1
        assert linked_prompts[0]["name"] == "test_prompt"
        assert linked_prompts[0]["version"] == "1"


@skip_when_testing_trace_sdk
def test_tracing_client_log_spans():
    with mlflow.start_run():
        # Create a trace
        with mlflow.start_span("test_span") as span:
            trace_id = mlflow.get_active_trace_id()

        # End the trace to make sure it's persisted

        # Create a test span using OTelReadableSpan
        trace_id_int = int(trace_id.replace("tr-", ""), 16)
        span_id_int = int("abc123def456", 16)

        readable_span = OTelReadableSpan(
            name="test_log_spans",
            context=SpanContext(
                trace_id=trace_id_int,
                span_id=span_id_int,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            ),
            parent=None,
            start_time=1000000000,  # 1 second in nanoseconds
            end_time=2000000000,  # 2 seconds in nanoseconds
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_id),
                "mlflow.spanInputs": '{"input": "test_input"}',
                "mlflow.spanOutputs": '{"output": "test_output"}',
                "mlflow.spanType": '"UNKNOWN"',
            },
            status=SpanStatus(SpanStatusCode.OK).to_otel_status(),
            resource=None,
            events=[],
        )

        # Create MLflow span from OpenTelemetry span
        span = create_mlflow_span(readable_span, trace_id)

        # Test logging the span using TracingClient
        client = TracingClient()
        logged_spans = asyncio.run(client.log_spans([span]))

        # Verify the returned spans are the same
        assert len(logged_spans) == 1
        assert logged_spans[0] == span
        assert logged_spans[0].trace_id == trace_id
        assert logged_spans[0].span_id == span.span_id
