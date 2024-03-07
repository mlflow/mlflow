import time
from unittest import mock


def test_trace():
    import mlflow

    # Hack
    from mlflow.traces.fluent import _TRACER_PROVIDER
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    client_mock = mock.MagicMock()
    _TRACER_PROVIDER.add_span_processor(SimpleSpanProcessor(client_mock))

    class MyModel:
        def __init__(self):
            self.x = 0

        @mlflow.trace(name="predict", attributes={"model_version": "1.0.0"})
        def predict(self, model_input, params=None):
            x = model_input["x"]
            y = model_input["y"]
            time.sleep(0.1)
            with mlflow.start_span(name="first_step") as span:
                x2 = x ** 2
                span.set_inputs({"x": x})
                span.set_outputs({"x2": x2})
                span.set_attribute("how", "cool")
                time.sleep(0.3)

            with mlflow.start_span(name="second_step", attributes={"this is": "second_step"}) as span:
                with mlflow.start_span(name="second_step_inner") as inner_span:
                    y2 = y ** 2
                    inner_span.set_inputs({"y": y})
                    inner_span.set_outputs({"y2": y2})
                    time.sleep(0.2)

                y3 = x2 * y2
                span.set_inputs({"x2": x2, "y2": y2})
                span.set_outputs({"y3": y3})
                span.set_attribute("how", "nice")
                time.sleep(0.5)

            return self.sum(x2, y3)

        @mlflow.trace(name="sum_step")
        def sum(self, a, b=0):
            time.sleep(1)
            return a + b

    model = MyModel()

    model.predict({"x": 1, "y": 2})

    assert client_mock.log_trace.call_count == 1
    captured_traces = [call[0][0] for call in client_mock.log_trace.call_args_list]
    assert len(captured_traces) == 1
    trace = captured_traces[0]
    trace.trace_info.trace_name == "predict"
    trace.trace_info.start_time is not None
    trace.trace_info.end_time is not None
    spans = trace.trace_data.spans
    assert len(spans) == 5
    root_span = spans[-1]
    assert root_span.name == "predict"
    assert root_span.start_time == trace.trace_info.start_time
    assert root_span.end_time == trace.trace_info.end_time
    assert root_span.inputs == {"model_input": {"x": 1, "y": 2}, "params": None}
    assert root_span.outputs == {"output": 6}
    assert dict(root_span.attributes) == {
        "function_name": "predict",
        "model_version": "1.0.0",
    }