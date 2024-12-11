"""
This example demonstrates how to create a trace to track the execution of a multi-threaded application.

To trace a multi-threaded operation, you need to use the low-level MLflow client APIs to create a trace and spans, because the high-level fluent APIs are not thread-safe.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import mlflow

exp = mlflow.set_experiment("mlflow-tracing-example")
exp_id = exp.experiment_id

# Initialize MLflow client.
client = mlflow.MlflowClient()


def task(x: int, request_id: str, parent_span_id: str) -> int:
    # Create a span for the task and connect it to the given parent span created by the main thread.
    print(f"Task {x} started")
    child_span = client.start_span(
        name="child_span",
        # Specify the ID of the trace and the parent span ID to which the child span belongs.
        request_id=request_id,
        parent_id=parent_span_id,
        # Each span has its own inputs.
        inputs={"x": x},
    )

    # Some long-running operation.
    y = x**2
    time.sleep(1)

    # End the child span.
    client.end_span(
        request_id=request_id,
        span_id=child_span.span_id,
        # Set the output(s) of the span.
        outputs=y,
    )

    print(f"Task {x} completed")
    return y


# Create the root span for the main thread.
root_span = client.start_trace(name="my_trace_multithreading")
request_id = root_span.request_id

xs = [1, 2, 3, 4, 5, 6]
with ThreadPoolExecutor(max_workers=2) as executor:
    # Create a task for each element in `xs`.
    # Each task is given the request ID and span ID of the root span,
    # so that it can connect the child span to the root span in the main thread.
    futures = [executor.submit(task, x, request_id, root_span.span_id) for x in xs]

    # Wait for all tasks to complete.
    results = [future.result() for future in futures]

# End the root span.
client.end_trace(request_id=request_id)

# Retrieve the just created trace.
trace = mlflow.get_last_active_trace()

# Print the trace in JSON format
print(trace.to_json(pretty=True))

# The trace should contain 7 spans in total: 1 root span and 6 child spans.
assert len(trace.data.spans) == len(xs) + 1

print(
    "\033[92m"
    + "🤖Now run `mlflow server` and open MLflow UI to see the trace visualization!"
    + "\033[0m"
)
