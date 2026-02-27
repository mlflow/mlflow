"""
This example demonstrates how to create a trace with multiple spans using the high-level MLflow fluent APIs.
"""

import mlflow

mlflow.set_experiment("mlflow-tracing-example")


# Decorating the function with `@mlflow.trace` decorator is the easiest way to trace your function.
# MLflow will create a trace for function calls and automatically
# captures function name, inputs, output, and more.
@mlflow.trace
def f1(x: int) -> int:
    return x + 1


# You can also specify additional metadata for the trace
@mlflow.trace(
    span_type="math",
    attributes={"operation": "addition"},
)
def f2(x: int) -> int:
    # MLflow keeps track of the call hierarchy. Calling `f1` inside
    # `f2` will create a child span `f1` under the `f2` span.
    x = f1(x) + 2

    # You can also create a span for an arbitrary block of code using `with mlflow.start_span` context manager.
    with mlflow.start_span(name="leaf", attributes={"operation": "exponentiation"}) as span:
        # Inputs and outputs need to be set explicitly for manually created spans.
        span.set_inputs({"x": x})
        x = x**2
        span.set_outputs({"x": x})

    return x


assert f2(1) == 16

# You can access the last trace via get_last_active_trace_id API.
trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id)

# Alternatively, you can use `search_traces` API to retrieve
# traces that meet certain criteria.
traces = mlflow.search_traces(
    filter_string="timestamp > 0",
    max_results=1,
)

# Print the trace in JSON format
print(trace.to_json(pretty=True))

print(
    "\033[92m"
    + "🤖Now run `mlflow server` and open MLflow UI to see the trace visualization!"
    + "\033[0m"
)
