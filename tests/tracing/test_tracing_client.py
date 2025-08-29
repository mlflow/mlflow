import json
import uuid

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
from mlflow.entities.span import create_mlflow_span
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.utils import TraceJSONEncoder

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
def test_tracing_client_get_trace_with_database_stored_spans():
    """Test that TracingClient.get_trace works with spans stored in the database."""
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

    client = TracingClient()

    with mlflow.start_run():
        experiment_id = mlflow.active_run().info.experiment_id
        trace_id = f"tr-{uuid.uuid4().hex}"

        store = client.store
        if not isinstance(store, SqlAlchemyStore):
            return

        otel_span = OTelReadableSpan(
            name="test_span",
            context=trace_api.SpanContext(
                trace_id=12345,
                span_id=111,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(1),
            ),
            parent=None,
            attributes={
                "mlflow.traceRequestId": json.dumps(trace_id, cls=TraceJSONEncoder),
                "llm.model_name": "test-model",
                "custom.attribute": "test-value",
            },
            start_time=1_000_000_000,
            end_time=2_000_000_000,
            resource=None,
        )

        span = create_mlflow_span(otel_span, trace_id, "LLM")

        store.log_spans(experiment_id, [span])

        trace = client.get_trace(trace_id)

        assert trace.info.trace_id == trace_id
        assert trace.info.tags.get(TraceTagKey.SPANS_LOCATION) == "tracking_store"

        assert len(trace.data.spans) == 1
        loaded_span = trace.data.spans[0]

        assert loaded_span.name == "test_span"
        assert loaded_span.trace_id == trace_id
        assert loaded_span.start_time_ns == 1_000_000_000
        assert loaded_span.end_time_ns == 2_000_000_000
        assert loaded_span.attributes.get("llm.model_name") == "test-model"
        assert loaded_span.attributes.get("custom.attribute") == "test-value"
