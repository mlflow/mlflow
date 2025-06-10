import json

import mlflow
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
