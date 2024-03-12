import time
from unittest import mock

import mlflow
from mlflow.tracing.types.model import SpanType, StatusCode


def test_trace(mock_client):
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(
            span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1}
        )
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    model = TestModel()
    model.predict(2, 5)

    mock_client.log_trace.assert_called_once()
    trace = mock_client.log_trace.call_args[0][0]
    trace_info = trace.trace_info
    assert trace_info.trace_id is not None
    assert trace_info.start_time <= trace_info.end_time - 0.1 * 1e9  # at least 0.1 sec
    assert trace_info.status.status_code == StatusCode.OK
    assert trace_info.inputs == '{"x": 2, "y": 5}'
    assert trace_info.outputs == '{"output": 64}'

    spans = trace.trace_data.spans
    assert len(spans) == 3

    span_name_to_span = {span.name: span for span in spans}
    root_span = span_name_to_span["predict"]
    assert root_span.start_time == trace_info.start_time
    assert root_span.end_time == trace_info.end_time
    assert root_span.parent_span_id is None
    assert root_span.inputs == {"x": 2, "y": 5}
    assert root_span.outputs == {"output": 64}
    assert root_span.attributes == {"function_name": "predict"}

    child_span_1 = span_name_to_span["add_one_with_custom_name"]
    assert child_span_1.parent_span_id == root_span.context.span_id
    assert child_span_1.inputs == {"z": 7}
    assert child_span_1.outputs == {"output": 8}
    assert child_span_1.attributes == {"function_name": "add_one", "delta": 1}

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_span_id == root_span.context.span_id
    assert child_span_2.inputs == {"t": 8}
    assert child_span_2.outputs == {"output": 64}
    assert child_span_2.start_time <= child_span_2.end_time - 0.1 * 1e9
    assert child_span_2.attributes == {"function_name": "square"}


def test_trace_handle_exception_during_prediction(mock_client):
    # This test is to make sure that the exception raised by the main prediction
    # logic is raised properly and the trace is still logged.
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return self.some_operation_raise_error(x, y)

        @mlflow.trace()
        def some_operation_raise_error(self, x, y):
            raise ValueError("Some error")

    model = TestModel()

    try:
        model.predict(2, 5)
    except ValueError:
        pass

    # Trace should be logged even if the function fails, with status code ERROR
    mock_client.log_trace.assert_called_once()
    trace = mock_client.log_trace.call_args[0][0]
    trace_info = trace.trace_info
    assert trace_info.trace_id is not None
    assert trace_info.status.status_code == StatusCode.ERROR
    assert trace_info.inputs == '{"x": 2, "y": 5}'
    assert trace_info.outputs == ""

    spans = trace.trace_data.spans
    assert len(spans) == 2


def test_trace_ignore_exception_from_tracing_logic(mock_client):
    # This test is to make sure that the main prediction logic is not affected
    # by the exception raised by the tracing logic.
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return x + y

    model = TestModel()

    # Exception during span creation: no-op span wrapper created and no trace is logged
    with mock.patch("mlflow.tracing.fluent._get_tracer",
                    side_effect=ValueError("Some error")):
        output = model.predict(2, 5)

    assert output == 7
    mock_client.log_trace.assert_not_called()

    # Exception during inspecting inputs: trace is logged without inputs field
    with mock.patch("mlflow.tracing.utils.inspect.signature",
                    side_effect=ValueError("Some error")):
        output = model.predict(2, 5)

    assert output == 7
    mock_client.log_trace.assert_called_once()
    trace = mock_client.log_trace.call_args[0][0]
    trace_info = trace.trace_info
    assert trace_info.inputs == ""
    assert trace_info.outputs == '{"output": 7}'


def test_start_span_context_manager(mock_client):
    class TestModel:
        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                with mlflow.start_span(name="child_span_1", span_type=SpanType.LLM) as child_span:
                    child_span.set_inputs({"z": z})
                    z = z + 2
                    child_span.set_outputs({"output": z})
                    child_span.set_attributes({"delta": 2})

                res = self.square(z)
                root_span.set_outputs({"output": res})
            return res

        def square(self, t):
            with mlflow.start_span(name="child_span_2") as span:
                span.set_inputs({"t": t})
                res = t**2
                time.sleep(0.1)
                span.set_outputs({"output": res})
                return res

    model = TestModel()
    model.predict(1, 2)

    mock_client.log_trace.assert_called_once()
    trace = mock_client.log_trace.call_args[0][0]
    trace_info = trace.trace_info
    assert trace_info.trace_id is not None
    assert trace_info.start_time <= trace_info.end_time - 0.1 * 1e9  # at least 0.1 sec
    assert trace_info.status.status_code == StatusCode.OK
    assert trace_info.inputs == '{"x": 1, "y": 2}'
    assert trace_info.outputs == '{"output": 25}'

    spans = trace.trace_data.spans
    assert len(spans) == 3

    span_name_to_span = {span.name: span for span in spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.start_time == trace_info.start_time
    assert root_span.end_time == trace_info.end_time
    assert root_span.parent_span_id is None
    assert root_span.span_type == SpanType.UNKNOWN
    assert root_span.inputs == {"x": 1, "y": 2}
    assert root_span.outputs == {"output": 25}

    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_span_id == root_span.context.span_id
    assert child_span_1.span_type == SpanType.LLM
    assert child_span_1.inputs == {"z": 3}
    assert child_span_1.outputs == {"output": 5}
    assert child_span_1.attributes == {"delta": 2}

    child_span_2 = span_name_to_span["child_span_2"]
    assert child_span_2.parent_span_id == root_span.context.span_id
    assert child_span_2.span_type == SpanType.UNKNOWN
    assert child_span_2.inputs == {"t": 5}
    assert child_span_2.outputs == {"output": 25}
    assert child_span_2.start_time <= child_span_2.end_time - 0.1 * 1e9
