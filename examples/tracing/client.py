"""
This example demonstrates how to create a trace with multiple spans using the low-level MLflow client APIs.
"""

import mlflow

exp = mlflow.set_experiment("mlflow-tracing-example")
exp_id = exp.experiment_id

# Initialize MLflow client.
client = mlflow.MlflowClient()


def run(x: int, y: int) -> int:
    # Create a trace. The `start_trace` API returns a root span of the trace.
    root_span = client.start_trace(
        name="my_trace",
        inputs={"x": x, "y": y},
        # Tags are key-value pairs associated with the trace.
        # You can update the tags later using `client.set_trace_tag` API.
        tags={
            "fruit": "apple",
            "vegetable": "carrot",
        },
    )

    z = x + y

    # Trace ID is a unique identifier for the trace. You will need this ID
    # to interact with the trace later using the MLflow client.
    trace_id = root_span.trace_id

    # Create a child span of the root span.
    child_span = client.start_span(
        name="child_span",
        # Specify the trace ID to which the child span belongs.
        trace_id=trace_id,
        # Also specify the ID of the parent span to build the span hierarchy.
        # You can access the span ID via `span_id` property of the span object.
        parent_id=root_span.span_id,
        # Each span has its own inputs.
        inputs={"z": z},
        # Attributes are key-value pairs associated with the span.
        attributes={
            "model": "my_model",
            "temperature": 0.5,
        },
    )

    z = z**2

    # End the child span. Please make sure to end the child span before ending the root span.
    client.end_span(
        trace_id=trace_id,
        span_id=child_span.span_id,
        # Set the output(s) of the span.
        outputs=z,
        # Set the completion status, such as "OK" (default), "ERROR", etc.
        status="OK",
    )

    z = z + 1

    # End the root span.
    client.end_trace(
        trace_id=trace_id,
        # Set the output(s) of the span.
        outputs=z,
    )

    return z


assert run(1, 2) == 10

# Retrieve the trace just created using get_last_active_trace_id() API.
trace_id = mlflow.get_last_active_trace_id()
trace = client.get_trace(trace_id)

# Alternatively, you can use search_traces() API
# to retrieve the traces from the tracking server.
trace = client.search_traces(experiment_ids=[exp_id])[0]
assert trace.info.tags["fruit"] == "apple"
assert trace.info.tags["vegetable"] == "carrot"

# Update the tags using set_trace_tag() and delete_trace_tag() APIs.
client.set_trace_tag(trace.info.trace_id, "fruit", "orange")
client.delete_trace_tag(trace.info.trace_id, "vegetable")

trace = client.get_trace(trace.info.trace_id)
assert trace.info.tags["fruit"] == "orange"
assert "vegetable" not in trace.info.tags

# Print the trace in JSON format
print(trace.to_json(pretty=True))

print(
    "\033[92m"
    + "🤖Now run `mlflow server` and open MLflow UI to see the trace visualization!"
    + "\033[0m"
)
