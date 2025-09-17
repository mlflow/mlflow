import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

import mlflow
from mlflow.entities import (
    SpanEvent,
    SpanStatusCode,
    SpanType,
    Trace,
    TraceData,
    TraceInfo,
)
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import (
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.destination import MlflowExperiment
from mlflow.tracing.export.inference_table import pop_trace
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import _get_tracer, set_destination
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import (
    create_test_trace_info,
    get_traces,
    purge_traces,
    skip_when_testing_trace_sdk,
)


class DefaultTestModel:
    @mlflow.trace()
    def predict(self, x, y):
        z = x + y
        z = self.add_one(z)
        z = mlflow.trace(self.square)(z)
        return z  # noqa: RET504

    @mlflow.trace(span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1})
    def add_one(self, z):
        return z + 1

    def square(self, t):
        res = t**2
        time.sleep(0.1)
        return res


class DefaultAsyncTestModel:
    @mlflow.trace()
    async def predict(self, x, y):
        z = x + y
        z = await self.add_one(z)
        z = await mlflow.trace(self.square)(z)
        return z  # noqa: RET504

    @mlflow.trace(span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1})
    async def add_one(self, z):
        return z + 1

    async def square(self, t):
        res = t**2
        time.sleep(0.1)
        return res


class StreamTestModel:
    @mlflow.trace(output_reducer=lambda x: sum(x))
    def predict_stream(self, x, y):
        z = x + y
        for i in range(z):
            yield i

        # Generator with a normal func
        for i in range(z):
            yield self.square(i)

        # Nested generator
        yield from self.generate_numbers(z)

    @mlflow.trace
    def square(self, t):
        time.sleep(0.1)
        return t**2

    # No output_reducer -> record the list of outputs
    @mlflow.trace
    def generate_numbers(self, z):
        for i in range(z):
            yield i


class AsyncStreamTestModel:
    @mlflow.trace(output_reducer=lambda x: sum(x))
    async def predict_stream(self, x, y):
        z = x + y
        for i in range(z):
            yield i

        # Generator with a normal func
        for i in range(z):
            yield await self.square(i)

        # Nested generator
        async for number in self.generate_numbers(z):
            yield number

    @mlflow.trace
    async def square(self, t):
        await asyncio.sleep(0.1)
        return t**2

    @mlflow.trace
    async def generate_numbers(self, z):
        for i in range(z):
            yield i


class ErroringTestModel:
    @mlflow.trace()
    def predict(self, x, y):
        return self.some_operation_raise_error(x, y)

    @mlflow.trace()
    def some_operation_raise_error(self, x, y):
        raise ValueError("Some error")


class ErroringAsyncTestModel:
    @mlflow.trace()
    async def predict(self, x, y):
        return await self.some_operation_raise_error(x, y)

    @mlflow.trace()
    async def some_operation_raise_error(self, x, y):
        raise ValueError("Some error")


class ErroringStreamTestModel:
    @mlflow.trace
    def predict_stream(self, x):
        for i in range(x):
            yield self.some_operation_raise_error(i)

    @mlflow.trace
    def some_operation_raise_error(self, i):
        if i >= 1:
            raise ValueError("Some error")
        return i


@pytest.fixture
def mock_client():
    client = mock.MagicMock()
    with mock.patch("mlflow.tracing.fluent.TracingClient", return_value=client):
        yield client


@pytest.mark.parametrize("with_active_run", [True, False])
@pytest.mark.parametrize("wrap_sync_func", [True, False])
def test_trace(wrap_sync_func, with_active_run, async_logging_enabled):
    model = DefaultTestModel() if wrap_sync_func else DefaultAsyncTestModel()

    if with_active_run:
        if IS_TRACING_SDK_ONLY:
            pytest.skip("Skipping test because mlflow or mlflow-skinny is not installed.")

        with mlflow.start_run() as run:
            model.predict(2, 5) if wrap_sync_func else asyncio.run(model.predict(2, 5))
            run_id = run.info.run_id
    else:
        model.predict(2, 5) if wrap_sync_func else asyncio.run(model.predict(2, 5))

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id is not None
    assert trace.info.experiment_id == _get_experiment_id()
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.state == TraceState.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "64"
    if with_active_run:
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response == "64"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["predict"]
    # TODO: Trace info timestamp is not accurate because it is not adjusted to exclude the latency
    # assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanFunctionName": "predict",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 2, "y": 5},
        "mlflow.spanOutputs": 64,
    }

    child_span_1 = span_name_to_span["add_one_with_custom_name"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 1,
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanFunctionName": "add_one",
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": {"z": 7},
        "mlflow.spanOutputs": 8,
    }

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanFunctionName": "square",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 8},
        "mlflow.spanOutputs": 64,
    }


@pytest.mark.parametrize("wrap_sync_func", [True, False])
def test_trace_stream(wrap_sync_func):
    model = StreamTestModel() if wrap_sync_func else AsyncStreamTestModel()

    stream = model.predict_stream(1, 2)

    # Trace should not be logged until the generator is consumed
    assert get_traces() == []
    # The span should not be set to active
    # because the generator is not yet consumed
    assert mlflow.get_current_active_span() is None

    chunks = []
    if wrap_sync_func:
        for chunk in stream:
            chunks.append(chunk)
            # The `predict` span should not be active here.
            assert mlflow.get_current_active_span() is None
    else:

        async def consume_stream():
            async for chunk in stream:
                chunks.append(chunk)
                assert mlflow.get_current_active_span() is None

        asyncio.run(consume_stream())

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id is not None
    assert trace.info.experiment_id == _get_experiment_id()
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == SpanStatusCode.OK
    metadata = trace.info.request_metadata
    assert metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert metadata[TraceMetadataKey.OUTPUTS] == "11"  # sum of the outputs

    assert len(trace.data.spans) == 5  # 1 root span + 3 square + 1 generate_numbers

    root_span = trace.data.spans[0]
    assert root_span.name == "predict_stream"
    assert root_span.inputs == {"x": 1, "y": 2}
    assert root_span.outputs == 11
    assert len(root_span.events) == 9
    assert root_span.events[0].name == "mlflow.chunk.item.0"
    assert root_span.events[0].attributes == {"mlflow.chunk.value": "0"}
    assert root_span.events[8].name == "mlflow.chunk.item.8"

    # Spans for the chid 'square' function
    for i in range(3):
        assert trace.data.spans[i + 1].name == f"square_{i + 1}"
        assert trace.data.spans[i + 1].inputs == {"t": i}
        assert trace.data.spans[i + 1].outputs == i**2
        assert trace.data.spans[i + 1].parent_id == root_span.span_id

    # Span for the 'generate_numbers' function
    assert trace.data.spans[4].name == "generate_numbers"
    assert trace.data.spans[4].inputs == {"z": 3}
    assert trace.data.spans[4].outputs == [0, 1, 2]  # list of outputs
    assert len(trace.data.spans[4].events) == 3


def test_trace_with_databricks_tracking_uri(databricks_tracking_uri, monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    model = DefaultTestModel()

    with (
        mock.patch(
            "mlflow.tracing.client.TracingClient._upload_trace_data"
        ) as mock_upload_trace_data,
        mock.patch("mlflow.tracing.client._get_store") as mock_get_store,
    ):
        model.predict(2, 5)
        mlflow.flush_trace_async_logging(terminate=True)

    mock_get_store().start_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()


# NB: async logging should be no-op for model serving,
# but we test it here to make sure it doesn't break
@skip_when_testing_trace_sdk
def test_trace_in_databricks_model_serving(
    mock_databricks_serving_with_tracing_env, async_logging_enabled
):
    # Dummy flask app for prediction
    import flask

    from mlflow.pyfunc.context import Context, set_prediction_context

    app = flask.Flask(__name__)

    @app.route("/invocations", methods=["POST"])
    def predict():
        data = json.loads(flask.request.data.decode("utf-8"))
        request_id = flask.request.headers.get("X-Request-ID")

        with set_prediction_context(Context(request_id=request_id)):
            prediction = TestModel().predict(**data)

        trace = pop_trace(request_id=request_id)

        result = json.dumps(
            {
                "prediction": prediction,
                "trace": trace,
            },
            default=str,
        )
        return flask.Response(response=result, status=200, mimetype="application/json")

    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            with mlflow.start_span(name="square") as span:
                z = self.square(z)
                span.add_event(SpanEvent("event", 0, attributes={"foo": "bar"}))
            return z

        @mlflow.trace(span_type=SpanType.LLM, name="custom", attributes={"delta": 1})
        def add_one(self, z):
            return z + 1

        def square(self, t):
            return t**2

    # Mimic scoring request
    databricks_request_id = "request-12345"
    response = app.test_client().post(
        "/invocations",
        headers={"X-Request-ID": databricks_request_id},
        data=json.dumps({"x": 2, "y": 5}),
    )

    assert response.status_code == 200
    assert response.json["prediction"] == 64

    trace_dict = response.json["trace"]
    trace = Trace.from_dict(trace_dict)
    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.client_request_id == databricks_request_id
    assert trace.info.request_metadata[TRACE_SCHEMA_VERSION_KEY] == "3"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["predict"]
    assert isinstance(root_span._trace_id, str)
    assert isinstance(root_span.span_id, str)
    assert isinstance(root_span.start_time_ns, int)
    assert isinstance(root_span.end_time_ns, int)
    assert root_span.status.status_code.value == "OK"
    assert root_span.status.description == ""
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": SpanType.UNKNOWN,
        "mlflow.spanFunctionName": "predict",
        "mlflow.spanInputs": {"x": 2, "y": 5},
        "mlflow.spanOutputs": 64,
    }
    assert root_span.events == []

    child_span_1 = span_name_to_span["custom"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 1,
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": SpanType.LLM,
        "mlflow.spanFunctionName": "add_one",
        "mlflow.spanInputs": {"z": 7},
        "mlflow.spanOutputs": 8,
    }
    assert child_span_1.events == []

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": SpanType.UNKNOWN,
    }
    assert asdict(child_span_2.events[0]) == {
        "name": "event",
        "timestamp": 0,
        "attributes": {"foo": "bar"},
    }

    # The trace should be removed from the buffer after being retrieved
    assert pop_trace(request_id=databricks_request_id) is None

    # In model serving, the traces should not be stored in the fluent API buffer
    traces = get_traces()
    assert len(traces) == 0


@skip_when_testing_trace_sdk
def test_trace_in_model_evaluation(monkeypatch, async_logging_enabled):
    from mlflow.pyfunc.context import Context, set_prediction_context

    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return x + y

    model = TestModel()

    # mock _upload_trace_data to avoid generating trace data file
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        request_id_1 = "tr-eval-123"
        with set_prediction_context(Context(request_id=request_id_1, is_evaluate=True)):
            model.predict(1, 2)

        request_id_2 = "tr-eval-456"
        with set_prediction_context(Context(request_id=request_id_2, is_evaluate=True)):
            model.predict(3, 4)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    trace = mlflow.get_trace(request_id_1)
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
    assert trace.info.request_metadata[TRACE_SCHEMA_VERSION_KEY] == str(TRACE_SCHEMA_VERSION)
    assert trace.info.tags[TraceTagKey.EVAL_REQUEST_ID] == request_id_1

    trace = mlflow.get_trace(request_id_2)
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
    assert trace.info.tags[TraceTagKey.EVAL_REQUEST_ID] == request_id_2


@pytest.mark.parametrize("sync", [True, False])
def test_trace_handle_exception_during_prediction(sync):
    # This test is to make sure that the exception raised by the main prediction
    # logic is raised properly and the trace is still logged.
    model = ErroringTestModel() if sync else ErroringAsyncTestModel()

    with pytest.raises(ValueError, match=r"Some error"):
        model.predict(2, 5) if sync else asyncio.run(model.predict(2, 5))

    # Trace should be logged even if the function fails, with status code ERROR
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.trace_id is not None
    assert trace.info.state == TraceState.ERROR
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == ""

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response is None
    assert len(trace.data.spans) == 2


def test_trace_handle_exception_during_streaming():
    model = ErroringStreamTestModel()

    stream = model.predict_stream(2)

    chunks = []
    with pytest.raises(ValueError, match=r"Some error"):  # noqa: PT012
        for chunk in stream:
            chunks.append(chunk)

    # The test model raises an error after the first chunk
    assert len(chunks) == 1

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.state == TraceState.ERROR
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2}'

    # The test model is expected to produce three spans
    # 1. Root span (error - inherited from the child)
    # 2. First chunk span (OK)
    # 3. Second chunk span (error)
    spans = trace.data.spans
    assert len(spans) == 3
    assert spans[0].name == "predict_stream"
    assert spans[0].status.status_code == SpanStatusCode.ERROR
    assert spans[1].name == "some_operation_raise_error_1"
    assert spans[1].status.status_code == SpanStatusCode.OK
    assert spans[2].name == "some_operation_raise_error_2"
    assert spans[2].status.status_code == SpanStatusCode.ERROR

    # One chunk event + one exception event
    assert len(spans[0].events) == 2
    assert spans[0].events[0].name == "mlflow.chunk.item.0"
    assert spans[0].events[1].name == "exception"


@pytest.mark.parametrize(
    "model",
    [
        DefaultTestModel(),
        DefaultAsyncTestModel(),
        StreamTestModel(),
        AsyncStreamTestModel(),
    ],
)
def test_trace_ignore_exception(monkeypatch, model):
    # This test is to make sure that the main prediction logic is not affected
    # by the exception raised by the tracing logic.
    def _call_model_and_assert_output(model):
        if isinstance(model, DefaultTestModel):
            output = model.predict(2, 5)
            assert output == 64
        elif isinstance(model, DefaultAsyncTestModel):
            output = asyncio.run(model.predict(2, 5))
            assert output == 64
        elif isinstance(model, StreamTestModel):
            stream = model.predict_stream(2, 5)
            assert len(list(stream)) == 21
        elif isinstance(model, AsyncStreamTestModel):
            astream = model.predict_stream(2, 5)

            async def _consume_stream():
                return [chunk async for chunk in astream]

            stream = asyncio.run(_consume_stream())
            assert len(list(stream)) == 21
        else:
            raise ValueError("Unknown model type")

    # Exception during starting span: trace should not be logged.
    with mock.patch("mlflow.tracing.provider._get_tracer", side_effect=ValueError("Some error")):
        _call_model_and_assert_output(model)

    assert get_traces() == []

    # Exception during ending span: trace should not be logged.
    tracer = _get_tracer(__name__)

    def _always_fail(*args, **kwargs):
        raise ValueError("Some error")

    monkeypatch.setattr(tracer.span_processor, "on_end", _always_fail)
    _call_model_and_assert_output(model)
    assert len(get_traces()) == 0


def test_trace_skip_resolving_unrelated_tags_to_traces():
    with mock.patch("mlflow.tracking.context.registry.DatabricksRepoRunContext") as mock_context:
        mock_context.in_context.return_value = ["unrelated tags"]

        model = DefaultTestModel()
        model.predict(2, 5)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert "unrelated tags" not in trace.info.tags


# Tracing SDK doesn't have `create_experiment` support
@skip_when_testing_trace_sdk
def test_trace_with_experiment_id():
    exp_1 = mlflow.create_experiment("exp_1")
    exp_2 = mlflow.set_experiment("exp_2").experiment_id  # active experiment

    @mlflow.trace(trace_destination=MlflowExperiment(exp_1))
    def predict_1():
        with mlflow.start_span(name="child_span"):
            return

    @mlflow.trace()
    def predict_2():
        pass

    predict_1()
    traces = get_traces(experiment_id=exp_1)
    assert len(traces) == 1
    assert traces[0].info.experiment_id == exp_1
    assert len(traces[0].data.spans) == 2
    assert get_traces(experiment_id=exp_2) == []

    predict_2()
    traces = get_traces(experiment_id=exp_2)
    assert len(traces) == 1
    assert traces[0].info.experiment_id == exp_2


# Tracing SDK doesn't have `create_experiment` support
@skip_when_testing_trace_sdk
def test_trace_with_experiment_id_issue_warning_when_not_root_span():
    exp_1 = mlflow.create_experiment("exp_1")

    @mlflow.trace(trace_destination=MlflowExperiment(exp_1))
    def predict_1():
        return predict_2()

    @mlflow.trace(trace_destination=MlflowExperiment(exp_1))
    def predict_2():
        return

    with mock.patch("mlflow.tracing.provider._logger") as mock_logger:
        predict_1()

    assert mock_logger.warning.call_count == 1
    assert mock_logger.warning.call_args[0][0] == (
        "The `experiment_id` parameter can only be used for root spans, but the span "
        "`predict_2` is not a root span. The specified value `1` will be ignored."
    )


def test_start_span_context_manager(async_logging_enabled):
    datetime_now = datetime.now()

    class TestModel:
        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                with mlflow.start_span(name="child_span", span_type=SpanType.LLM) as child_span:
                    child_span.set_inputs(z)
                    z = z + 2
                    child_span.set_outputs(z)
                    child_span.set_attributes({"delta": 2, "time": datetime_now})

                res = self.square(z)
                root_span.set_outputs(res)
            return res

        def square(self, t):
            with mlflow.start_span(name="child_span") as span:
                span.set_inputs({"t": t})
                res = t**2
                time.sleep(0.1)
                span.set_outputs(res)
                return res

    model = TestModel()
    model.predict(1, 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id is not None
    assert trace.info.experiment_id == _get_experiment_id()
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.state == TraceState.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "25"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "25"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": 25,
    }

    # Span with duplicate name should be renamed to have an index number like "_1", "_2", ...
    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 2,
        "time": str(datetime_now),
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }

    child_span_2 = span_name_to_span["child_span_2"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 5},
        "mlflow.spanOutputs": 25,
    }
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6


def test_start_span_context_manager_with_imperative_apis(async_logging_enabled):
    # This test is to make sure that the spans created with fluent APIs and imperative APIs
    # (via MLflow client) are correctly linked together. This usage is not recommended but
    # should be supported for the advanced use cases like using LangChain callbacks as a
    # part of broader tracing.
    class TestModel:
        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                child_span = start_span_no_context(
                    name="child_span_1",
                    span_type=SpanType.LLM,
                    parent_span=root_span,
                )
                child_span.set_inputs(z)

                z = z + 2
                time.sleep(0.1)

                child_span.set_outputs(z)
                child_span.set_attributes({"delta": 2})
                child_span.end()

                root_span.set_outputs(z)
            return z

    model = TestModel()
    model.predict(1, 2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.trace_id is not None
    assert trace.info.experiment_id == _get_experiment_id()
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.state == TraceState.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "5"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "5"
    assert len(trace.data.spans) == 2

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": 5,
    }

    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 2,
        "mlflow.traceRequestId": trace.info.trace_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }


def test_mlflow_trace_isolated_from_other_otel_processors():
    # Set up non-MLFlow tracer
    import opentelemetry.sdk.trace as trace_sdk
    from opentelemetry import trace

    class MockOtelExporter(trace_sdk.export.SpanExporter):
        def __init__(self):
            self.exported_spans = []

        def export(self, spans):
            self.exported_spans.extend(spans)

    other_exporter = MockOtelExporter()
    provider = trace_sdk.TracerProvider()
    processor = trace_sdk.export.SimpleSpanProcessor(other_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Create MLflow trace
    with mlflow.start_span(name="mlflow_span"):
        pass

    # Create non-MLflow trace
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("non_mlflow_span"):
        pass

    # MLflow only processes spans created with MLflow APIs
    assert len(get_traces()) == 1
    assert mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans[0].name == "mlflow_span"

    # Other spans are processed by the other processor
    assert len(other_exporter.exported_spans) == 1
    assert other_exporter.exported_spans[0].name == "non_mlflow_span"


@mock.patch("mlflow.tracing.display.get_display_handler")
def test_get_trace(mock_get_display_handler):
    model = DefaultTestModel()
    model.predict(2, 5)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_id = trace.info.trace_id
    mock_get_display_handler.reset_mock()

    # Fetch trace from in-memory buffer
    trace_in_memory = mlflow.get_trace(trace_id)
    assert trace.info.trace_id == trace_in_memory.info.trace_id
    mock_get_display_handler.assert_not_called()

    # Fetch trace from backend
    trace_from_backend = mlflow.get_trace(trace.info.trace_id)
    assert trace.info.trace_id == trace_from_backend.info.trace_id
    mock_get_display_handler.assert_not_called()

    # If not found, return None with warning
    with mock.patch("mlflow.tracing.fluent._logger") as mock_logger:
        assert mlflow.get_trace("not_found") is None
        mock_logger.warning.assert_called_once()


def test_test_search_traces_empty(mock_client):
    mock_client.search_traces.return_value = PagedList([], token=None)

    traces = mlflow.search_traces()
    assert len(traces) == 0

    if not IS_TRACING_SDK_ONLY:
        default_columns = Trace.pandas_dataframe_columns()
        assert traces.columns.tolist() == default_columns

        traces = mlflow.search_traces(extract_fields=["foo.inputs.bar"])
        assert traces.columns.tolist() == [*default_columns, "foo.inputs.bar"]

        mock_client.search_traces.assert_called()


@pytest.mark.parametrize("return_type", ["pandas", "list"])
def test_search_traces(return_type, mock_client):
    if return_type == "pandas" and IS_TRACING_SDK_ONLY:
        pytest.skip("Skipping test because mlflow or mlflow-skinny is not installed.")

    mock_client.search_traces.return_value = PagedList(
        [
            Trace(
                info=create_test_trace_info(f"tr-{i}"),
                data=TraceData([]),
            )
            for i in range(10)
        ],
        token=None,
    )

    traces = mlflow.search_traces(
        experiment_ids=["1"],
        filter_string="name = 'foo'",
        max_results=10,
        order_by=["timestamp DESC"],
        return_type=return_type,
    )

    if return_type == "pandas":
        import pandas as pd

        assert isinstance(traces, pd.DataFrame)
    else:
        assert isinstance(traces, list)
        assert all(isinstance(trace, Trace) for trace in traces)

    assert len(traces) == 10
    mock_client.search_traces.assert_called_once_with(
        experiment_ids=["1"],
        run_id=None,
        filter_string="name = 'foo'",
        max_results=10,
        order_by=["timestamp DESC"],
        page_token=None,
        model_id=None,
        sql_warehouse_id=None,
        include_spans=True,
    )


def test_search_traces_invalid_return_types(mock_client):
    with pytest.raises(MlflowException, match=r"Invalid return type"):
        mlflow.search_traces(return_type="invalid")

    with pytest.raises(MlflowException, match=r"The `extract_fields`"):
        mlflow.search_traces(extract_fields=["foo.inputs.bar"], return_type="list")


def test_search_traces_with_pagination(mock_client):
    traces = [
        Trace(
            info=create_test_trace_info(f"tr-{i}"),
            data=TraceData([]),
        )
        for i in range(30)
    ]

    mock_client.search_traces.side_effect = [
        PagedList(traces[:10], token="token-1"),
        PagedList(traces[10:20], token="token-2"),
        PagedList(traces[20:], token=None),
    ]

    traces = mlflow.search_traces(experiment_ids=["1"])

    assert len(traces) == 30
    common_args = {
        "experiment_ids": ["1"],
        "run_id": None,
        "max_results": SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        "filter_string": None,
        "order_by": None,
        "include_spans": True,
        "model_id": None,
        "sql_warehouse_id": None,
    }
    mock_client.search_traces.assert_has_calls(
        [
            mock.call(**common_args, page_token=None),
            mock.call(**common_args, page_token="token-1"),
            mock.call(**common_args, page_token="token-2"),
        ]
    )


def test_search_traces_with_default_experiment_id(mock_client):
    mock_client.search_traces.return_value = PagedList([], token=None)
    with mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="123"):
        mlflow.search_traces()

    mock_client.search_traces.assert_called_once_with(
        experiment_ids=["123"],
        run_id=None,
        filter_string=None,
        max_results=SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by=None,
        page_token=None,
        model_id=None,
        sql_warehouse_id=None,
        include_spans=True,
    )


@skip_when_testing_trace_sdk
def test_search_traces_yields_expected_dataframe_contents(monkeypatch):
    model = DefaultTestModel()
    expected_traces = []
    for _ in range(10):
        model.predict(2, 5)
        time.sleep(0.1)

        trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
        expected_traces.append(trace)

    df = mlflow.search_traces(max_results=10, order_by=["timestamp ASC"])
    assert df.columns.tolist() == [
        "trace_id",
        "trace",
        "client_request_id",
        "state",
        "request_time",
        "execution_duration",
        "request",
        "response",
        "trace_metadata",
        "tags",
        "spans",
        "assessments",
    ]
    for idx, trace in enumerate(expected_traces):
        assert df.iloc[idx].trace_id == trace.info.trace_id
        assert Trace.from_json(df.iloc[idx].trace).info.trace_id == trace.info.trace_id
        assert df.iloc[idx].client_request_id == trace.info.client_request_id
        assert df.iloc[idx].state == trace.info.state
        assert df.iloc[idx].request_time == trace.info.request_time
        assert df.iloc[idx].execution_duration == trace.info.execution_duration
        assert df.iloc[idx].request == json.loads(trace.data.request)
        assert df.iloc[idx].response == json.loads(trace.data.response)
        assert df.iloc[idx].trace_metadata == trace.info.trace_metadata
        assert df.iloc[idx].spans == [s.to_dict() for s in trace.data.spans]
        assert df.iloc[idx].tags == trace.info.tags
        assert df.iloc[idx].assessments == trace.info.assessments


@skip_when_testing_trace_sdk
def test_search_traces_handles_missing_response_tags_and_metadata(mock_client):
    mock_client.search_traces.return_value = PagedList(
        [
            Trace(
                info=TraceInfo(
                    trace_id="5",
                    trace_location=TraceLocation.from_experiment_id("test"),
                    request_time=1,
                    execution_duration=2,
                    state=TraceState.OK,
                ),
                data=TraceData(spans=[]),
            )
        ],
        token=None,
    )

    df = mlflow.search_traces()
    assert df["response"].isnull().all()
    assert df["tags"].tolist() == [{}]
    # trace_metadata should only contain the schema version
    assert df["trace_metadata"].tolist() == [{TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)}]


@skip_when_testing_trace_sdk
def test_search_traces_extracts_fields_as_expected():
    model = DefaultTestModel()
    model.predict(2, 5)

    df = mlflow.search_traces(
        extract_fields=["predict.inputs.x", "predict.outputs", "add_one_with_custom_name.inputs.z"]
    )
    assert df["predict.inputs.x"].tolist() == [2]
    assert df["predict.outputs"].tolist() == [64]
    assert df["add_one_with_custom_name.inputs.z"].tolist() == [7]


# no spans have the input or output with name,
# some span has an input but we're looking for output,
@skip_when_testing_trace_sdk
def test_search_traces_with_input_and_no_output():
    with mlflow.start_span(name="with_input_and_no_output") as span:
        span.set_inputs({"a": 1})

    df = mlflow.search_traces(
        extract_fields=["with_input_and_no_output.inputs.a", "with_input_and_no_output.outputs"]
    )
    assert df["with_input_and_no_output.inputs.a"].tolist() == [1]
    assert df["with_input_and_no_output.outputs"].isnull().all()


@skip_when_testing_trace_sdk
def test_search_traces_with_non_dict_span_inputs_outputs():
    with mlflow.start_span(name="non_dict_span") as span:
        span.set_inputs(["a", "b"])
        span.set_outputs([1, 2, 3])

    df = mlflow.search_traces(
        extract_fields=["non_dict_span.inputs", "non_dict_span.outputs", "non_dict_span.inputs.x"]
    )
    assert df["non_dict_span.inputs"].tolist() == [["a", "b"]]
    assert df["non_dict_span.outputs"].tolist() == [[1, 2, 3]]
    assert df["non_dict_span.inputs.x"].isnull().all()


@skip_when_testing_trace_sdk
def test_search_traces_with_multiple_spans_with_same_name():
    class TestModel:
        @mlflow.trace(name="duplicate_name")
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(span_type=SpanType.LLM, name="duplicate_name", attributes={"delta": 1})
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    model = TestModel()
    model.predict(2, 5)

    df = mlflow.search_traces(
        extract_fields=[
            "duplicate_name.inputs.y",
            "duplicate_name.inputs.x",
            "duplicate_name.inputs.z",
            "duplicate_name_1.inputs.x",
            "duplicate_name_1.inputs.y",
            "duplicate_name_2.inputs.z",
        ]
    )
    # Duplicate spans would all be null
    assert df["duplicate_name.inputs.y"].isnull().all()
    assert df["duplicate_name.inputs.x"].isnull().all()
    assert df["duplicate_name.inputs.z"].isnull().all()
    assert df["duplicate_name_1.inputs.x"].tolist() == [2]
    assert df["duplicate_name_1.inputs.y"].tolist() == [5]
    assert df["duplicate_name_2.inputs.z"].tolist() == [7]


# Test a field that doesn't exist for extraction - we shouldn't throw, just return empty column
@skip_when_testing_trace_sdk
def test_search_traces_with_non_existent_field():
    model = DefaultTestModel()
    model.predict(2, 5)

    df = mlflow.search_traces(
        extract_fields=[
            "predict.inputs.k",
            "predict.inputs.x",
            "predict.outputs",
            "add_one_with_custom_name.inputs.z",
        ]
    )
    assert df["predict.inputs.k"].isnull().all()
    assert df["predict.inputs.x"].tolist() == [2]
    assert df["predict.outputs"].tolist() == [64]
    assert df["add_one_with_custom_name.inputs.z"].tolist() == [7]


@skip_when_testing_trace_sdk
def test_search_traces_span_and_field_name_with_dot():
    with mlflow.start_span(name="span.name") as span:
        span.set_inputs({"a.b": 0})
        span.set_outputs({"x.y": 1})

    df = mlflow.search_traces(
        extract_fields=[
            "`span.name`.inputs",
            "`span.name`.inputs.`a.b`",
            "`span.name`.outputs",
            "`span.name`.outputs.`x.y`",
        ]
    )

    assert df["span.name.inputs"].tolist() == [{"a.b": 0}]
    assert df["span.name.inputs.a.b"].tolist() == [0]
    assert df["span.name.outputs"].tolist() == [{"x.y": 1}]
    assert df["span.name.outputs.x.y"].tolist() == [1]


@skip_when_testing_trace_sdk
def test_search_traces_with_run_id():
    def _create_trace(name, tags=None):
        with mlflow.start_span(name=name) as span:
            for k, v in (tags or {}).items():
                mlflow.set_trace_tag(trace_id=span.request_id, key=k, value=v)
        return span.request_id

    def _get_names(traces):
        tags = traces["tags"].tolist()
        return [tags[i].get(TraceTagKey.TRACE_NAME) for i in range(len(tags))]

    with mlflow.start_run() as run1:
        _create_trace(name="tr-1")
        _create_trace(name="tr-2", tags={"fruit": "apple"})

    with mlflow.start_run() as run2:
        _create_trace(name="tr-3")
        _create_trace(name="tr-4", tags={"fruit": "banana"})
        _create_trace(name="tr-5", tags={"fruit": "apple"})

    traces = mlflow.search_traces()
    assert _get_names(traces) == ["tr-5", "tr-4", "tr-3", "tr-2", "tr-1"]

    traces = mlflow.search_traces(run_id=run1.info.run_id)
    assert _get_names(traces) == ["tr-2", "tr-1"]

    traces = mlflow.search_traces(
        run_id=run2.info.run_id,
        filter_string="tag.fruit = 'apple'",
    )
    assert _get_names(traces) == ["tr-5"]

    with pytest.raises(MlflowException, match="You cannot filter by run_id when it is already"):
        mlflow.search_traces(
            run_id=run2.info.run_id,
            filter_string="metadata.mlflow.sourceRun = '123'",
        )

    with pytest.raises(MlflowException, match=f"Run {run1.info.run_id} belongs to"):
        mlflow.search_traces(run_id=run1.info.run_id, experiment_ids=["1"])


@pytest.mark.parametrize(
    "extract_fields",
    [
        ["span.llm.inputs"],
        ["span.llm.inputs.x"],
        ["span.llm.outputs"],
    ],
)
@skip_when_testing_trace_sdk
def test_search_traces_invalid_extract_fields(extract_fields):
    with pytest.raises(MlflowException, match="Invalid field type"):
        mlflow.search_traces(extract_fields=extract_fields)


def test_get_last_active_trace_id():
    assert mlflow.get_last_active_trace_id() is None

    @mlflow.trace()
    def predict(x, y):
        return x + y

    predict(1, 2)
    predict(2, 5)
    predict(3, 6)

    trace_id = mlflow.get_last_active_trace_id()
    trace = mlflow.get_trace(trace_id)
    assert trace.info.trace_id is not None
    assert trace.data.request == '{"x": 3, "y": 6}'

    # Mutation of the copy should not affect the original trace logged in the backend
    trace.info.state = TraceState.ERROR
    original_trace = mlflow.get_trace(trace.info.trace_id)
    assert original_trace.info.state == TraceState.OK


def test_get_last_active_trace_thread_local():
    assert mlflow.get_last_active_trace_id() is None

    def run(id):
        @mlflow.trace(name=f"predict_{id}")
        def predict(x, y):
            return x + y

        predict(1, 2)

        return mlflow.get_last_active_trace_id(thread_local=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run, i) for i in range(10)]
        trace_ids = [future.result() for future in futures]

    assert len(trace_ids) == 10
    for i, trace_id in enumerate(trace_ids):
        trace = mlflow.get_trace(trace_id)
        assert trace.info.state == TraceState.OK
        assert trace.data.spans[0].name == f"predict_{i}"


def test_trace_with_classmethod():
    class TestModel:
        @mlflow.trace
        @classmethod
        def predict(cls, x, y):
            return x + y

    # Call the classmethod
    result = TestModel.predict(1, 2)
    assert result == 3

    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None

    trace = mlflow.get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0

    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == 3


def test_trace_with_classmethod_order_reversed():
    class TestModel:
        @classmethod
        @mlflow.trace
        def predict(cls, x, y):
            return x + y

    # Call the classmethod
    result = TestModel.predict(1, 2)
    assert result == 3

    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None

    trace = mlflow.get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0

    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == 3


def test_trace_with_staticmethod():
    class TestModel:
        @mlflow.trace
        @staticmethod
        def predict(x, y):
            return x + y

    # Call the staticmethod
    result = TestModel.predict(1, 2)
    assert result == 3

    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None

    trace = mlflow.get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0

    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == 3


def test_trace_with_staticmethod_order_reversed():
    class TestModel:
        @staticmethod
        @mlflow.trace
        def predict(x, y):
            return x + y

    # Call the staticmethod
    result = TestModel.predict(1, 2)
    assert result == 3

    # Get the last trace and verify inputs and outputs
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None

    trace = mlflow.get_trace(trace_id)
    assert trace is not None
    assert len(trace.data.spans) > 0

    # The first span should be our traced function
    span = trace.data.spans[0]
    assert span.name == "predict"
    assert span.inputs == {"x": 1, "y": 2}
    assert span.outputs == 3


def test_update_current_trace():
    @mlflow.trace(name="root_function")
    def f(x):
        mlflow.update_current_trace(tags={"fruit": "apple", "animal": "dog"})
        return g(x) + 1

    @mlflow.trace(name="level_1_function")
    def g(y):
        with mlflow.start_span(name="level_2_span"):
            mlflow.update_current_trace(tags={"fruit": "orange", "vegetable": "carrot"})
            return h(y) * 2

    @mlflow.trace(name="level_3_function")
    def h(z):
        with mlflow.start_span(name="level_4_span"):
            with mlflow.start_span(name="level_5_span"):
                mlflow.update_current_trace(tags={"depth": "deep", "level": "5"})
                return z + 10

    f(1)

    expected_tags = {
        "animal": "dog",
        "fruit": "orange",
        "vegetable": "carrot",
        "depth": "deep",
        "level": "5",
    }

    # Validate in-memory trace
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.state == TraceState.OK
    tags = {k: v for k, v in trace.info.tags.items() if not k.startswith("mlflow.")}
    assert tags == expected_tags

    # Validate backend trace
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.state == TraceState.OK
    tags = {k: v for k, v in traces[0].info.tags.items() if not k.startswith("mlflow.")}
    assert tags == expected_tags

    # Verify trace can be searched by span names (only when database backend is available)
    if not IS_TRACING_SDK_ONLY:
        trace_by_root_span = mlflow.search_traces(
            filter_string='span.name = "root_function"', return_type="list"
        )
        assert len(trace_by_root_span) == 1

        trace_by_level_2_span = mlflow.search_traces(
            filter_string='span.name = "level_2_span"', return_type="list"
        )
        assert len(trace_by_level_2_span) == 1

        trace_by_level_5_span = mlflow.search_traces(
            filter_string='span.name = "level_5_span"', return_type="list"
        )
        assert len(trace_by_level_5_span) == 1

        # All searches should return the same trace
        assert trace_by_root_span[0].info.request_id == trace.info.request_id
        assert trace_by_level_2_span[0].info.request_id == trace.info.request_id
        assert trace_by_level_5_span[0].info.request_id == trace.info.request_id


def test_update_current_trace_with_client_request_id():
    """Test that update_current_trace correctly handles client_request_id parameter."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    # Test updating during span execution
    with mlflow.start_span("test_span") as span:
        # Update with both tags and client_request_id
        mlflow.update_current_trace(tags={"operation": "test"}, client_request_id="req-12345")

        # Check in-memory trace during execution
        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            assert trace.info.client_request_id == "req-12345"
            tags = {k: v for k, v in trace.info.tags.items() if not k.startswith("mlflow.")}
            assert tags["operation"] == "test"

    # Test with tags only
    with mlflow.start_span("test_span_2") as span:
        mlflow.update_current_trace(tags={"operation": "tags_only"})

        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            assert trace.info.client_request_id is None
            tags = {k: v for k, v in trace.info.tags.items() if not k.startswith("mlflow.")}
            assert tags["operation"] == "tags_only"

    # Test with client_request_id only
    with mlflow.start_span("test_span_3") as span:
        mlflow.update_current_trace(client_request_id="req-67890")

        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            assert trace.info.client_request_id == "req-67890"


def test_update_current_trace_client_request_id_overwrites():
    """Test that client_request_id can be overwritten by subsequent calls."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    with mlflow.start_span("overwrite_test") as span:
        # First set
        mlflow.update_current_trace(client_request_id="req-initial")

        # Overwrite with new value
        mlflow.update_current_trace(client_request_id="req-updated")

        # Check during execution
        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            # Should have the updated value, not the initial one
            assert trace.info.client_request_id == "req-updated"


def test_update_current_trace_client_request_id_stringification():
    """Test that client_request_id is stringified when it's not a string."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    test_cases = [
        (123, "123"),
        (45.67, "45.67"),
        (True, "True"),
        (False, "False"),
        (None, None),  # None should remain None
        (["list", "value"], "['list', 'value']"),
        ({"dict": "value"}, "{'dict': 'value'}"),
    ]

    for input_value, expected_output in test_cases:
        with mlflow.start_span(f"stringification_test_{input_value}") as span:
            if input_value is None:
                # None should not update the client_request_id
                mlflow.update_current_trace(client_request_id=input_value)
                trace_manager = InMemoryTraceManager.get_instance()
                with trace_manager.get_trace(span.trace_id) as trace:
                    assert trace.info.client_request_id is None
            else:
                mlflow.update_current_trace(client_request_id=input_value)
                trace_manager = InMemoryTraceManager.get_instance()
                with trace_manager.get_trace(span.trace_id) as trace:
                    assert trace.info.client_request_id == expected_output
                    assert isinstance(trace.info.client_request_id, str)


def test_update_current_trace_with_metadata():
    """Test that update_current_trace correctly handles metadata parameter."""

    @mlflow.trace
    def f():
        mlflow.update_current_trace(
            metadata={
                "mlflow.source.name": "inference.py",
                "mlflow.source.git.commit": "1234567890",
                "mlflow.source.git.repoURL": "https://github.com/mlflow/mlflow",
                "non-string-metadata": 123,
            },
        )

    f()

    expected_metadata = {
        "mlflow.source.name": "inference.py",
        "mlflow.source.git.commit": "1234567890",
        "mlflow.source.git.repoURL": "https://github.com/mlflow/mlflow",
        "non-string-metadata": "123",  # Should be stringified
    }

    # Validate in-memory trace
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    for k, v in expected_metadata.items():
        assert trace.info.trace_metadata[k] == v

    # Validate backend trace
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    for k, v in expected_metadata.items():
        assert traces[0].info.trace_metadata[k] == v


@skip_when_testing_trace_sdk
def test_update_current_trace_should_not_raise_during_model_logging():
    """
    Tracing is disabled while model logging. When the model includes
    `update_current_trace` call, it should be no-op.
    """

    class MyModel(mlflow.pyfunc.PythonModel):
        @mlflow.trace
        def predict(self, model_inputs):
            mlflow.update_current_trace(tags={"fruit": "apple"})
            return [model_inputs[0] + 1]

    model = MyModel()

    model.predict([1])
    trace = get_traces()[0]
    assert trace.info.state == "OK"
    assert trace.info.tags["fruit"] == "apple"
    purge_traces()

    model_info = mlflow.pyfunc.log_model(
        python_model=model,
        name="model",
        input_example=[0],
    )
    # Trace should not be generated while logging the model
    assert get_traces() == []

    # Signature should be inferred properly without raising any exception
    assert model_info.signature is not None
    assert model_info.signature.inputs is not None
    assert model_info.signature.outputs is not None

    # Loading back the model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_model.predict([1])
    trace = get_traces()[0]
    assert trace.info.status == "OK"
    assert trace.info.tags["fruit"] == "apple"


def test_update_current_trace_with_state():
    """Test the state parameter in update_current_trace."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    # Test with TraceState enum
    with mlflow.start_span("test_span") as span:
        mlflow.update_current_trace(state=TraceState.ERROR)

        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            assert trace.info.state == TraceState.ERROR

    # Test with string state
    with mlflow.start_span("test_span_2") as span:
        mlflow.update_current_trace(state="OK")

        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            assert trace.info.state == TraceState.OK

    # Test with combined parameters
    with mlflow.start_span("test_span_3") as span:
        mlflow.update_current_trace(
            state="ERROR", tags={"error_type": "validation"}, client_request_id="req-123"
        )

        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            assert trace.info.state == TraceState.ERROR
            assert trace.info.tags["error_type"] == "validation"
            assert trace.info.client_request_id == "req-123"


def test_update_current_trace_state_none():
    """Test that state=None doesn't change trace state."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    with mlflow.start_span("test_span") as span:
        # First set state to OK
        mlflow.update_current_trace(state="OK")

        # Then call with state=None - should not change state
        mlflow.update_current_trace(state=None, tags={"test": "value"})

        trace_manager = InMemoryTraceManager.get_instance()
        with trace_manager.get_trace(span.trace_id) as trace:
            assert trace.info.state == TraceState.OK
            assert trace.info.tags["test"] == "value"


def test_update_current_trace_state_validation():
    """Test that state validation only allows OK or ERROR."""
    with mlflow.start_span("test_span"):
        # Valid states should work
        mlflow.update_current_trace(state="OK")
        mlflow.update_current_trace(state="ERROR")
        mlflow.update_current_trace(state=TraceState.OK)
        mlflow.update_current_trace(state=TraceState.ERROR)

        # Invalid string state should raise an exception
        with pytest.raises(
            MlflowException, match=r"State must be either 'OK' or 'ERROR', but got 'IN_PROGRESS'"
        ):
            mlflow.update_current_trace(state="IN_PROGRESS")

        # Invalid enum state should raise an exception
        with pytest.raises(
            MlflowException,
            match=r"State must be either 'OK' or 'ERROR', but got 'STATE_UNSPECIFIED'",
        ):
            mlflow.update_current_trace(state=TraceState.STATE_UNSPECIFIED)

        # Custom invalid string should raise an exception
        with pytest.raises(
            MlflowException, match=r"State must be either 'OK' or 'ERROR', but got 'CUSTOM_STATE'"
        ):
            mlflow.update_current_trace(state="CUSTOM_STATE")

        # Invalid types should raise an exception with a proper error message
        with pytest.raises(
            MlflowException, match=r"State must be either 'OK' or 'ERROR', but got '123'"
        ):
            mlflow.update_current_trace(state=123)


def test_span_record_exception_with_string():
    """Test record_exception method with string parameter."""
    with mlflow.start_span("test_span") as span:
        span.record_exception("Something went wrong")

    # Check persisted trace
    trace = get_traces()[0]
    spans = trace.data.spans
    test_span = spans[0]

    # Verify span status is ERROR
    assert test_span.status.status_code == SpanStatusCode.ERROR

    # Verify exception event was added
    exception_events = [event for event in test_span.events if "exception" in event.name.lower()]
    assert len(exception_events) == 1

    # Verify exception message is in the event
    exception_event = exception_events[0]
    assert "Something went wrong" in str(exception_event.attributes)


def test_span_record_exception_with_exception():
    """Test record_exception method with Exception parameter."""
    test_exception = ValueError("Custom error message")

    with mlflow.start_span("test_span") as span:
        span.record_exception(test_exception)

    # Check persisted trace
    trace = get_traces()[0]
    spans = trace.data.spans
    test_span = spans[0]

    # Verify span status is ERROR
    assert test_span.status.status_code == SpanStatusCode.ERROR

    # Verify exception event was added with proper exception details
    exception_events = [event for event in test_span.events if "exception" in event.name.lower()]
    assert len(exception_events) == 1

    exception_event = exception_events[0]
    event_attrs = str(exception_event.attributes)
    assert "ValueError" in event_attrs
    assert "Custom error message" in event_attrs


def test_span_record_exception_invalid_type():
    """Test record_exception method with invalid parameter type."""
    with mlflow.start_span("test_span") as span:
        with pytest.raises(
            MlflowException,
            match="The `exception` parameter must be an Exception instance or a string",
        ):
            span.record_exception(123)


def test_combined_state_and_record_exception():
    """Test using both status update and record_exception together."""

    @mlflow.trace
    def test_function():
        # Get current span and record exception
        span = mlflow.get_current_active_span()
        span.record_exception("Processing failed")

        # Update trace state independently
        mlflow.update_current_trace(state="ERROR", tags={"error_source": "processing"})
        return "result"

    test_function()

    # Check the trace
    trace = get_traces()[0]

    # Verify trace state was set to ERROR
    assert trace.info.state == TraceState.ERROR
    assert trace.info.tags["error_source"] == "processing"

    # Verify span has exception event and ERROR state
    spans = trace.data.spans
    root_span = spans[0]
    assert root_span.status.status_code == SpanStatusCode.ERROR

    exception_events = [event for event in root_span.events if "exception" in event.name.lower()]
    assert len(exception_events) == 1
    assert "Processing failed" in str(exception_events[0].attributes)


def test_span_record_exception_no_op_span():
    """Test that record_exception works gracefully with NoOpSpan."""
    # This should not raise an exception
    from mlflow.entities.span import NoOpSpan

    no_op_span = NoOpSpan()
    no_op_span.record_exception("This should be ignored")

    # Should not create any traces
    assert get_traces() == []


def test_update_current_trace_state_isolation():
    """Test that state update doesn't affect span status."""
    with mlflow.start_span("test_span") as span:
        # Set span status to OK explicitly
        span.set_status("OK")

        # Update trace state to ERROR
        mlflow.update_current_trace(state="ERROR")

        # Span status should still be OK
        assert span.status.status_code == SpanStatusCode.OK

    # Check the final persisted trace
    trace = get_traces()[0]
    assert trace.info.state == TraceState.ERROR

    # Verify span status remained OK despite trace state being ERROR
    spans = trace.data.spans
    test_span = spans[0]
    assert test_span.status.status_code == SpanStatusCode.OK


@skip_when_testing_trace_sdk
def test_non_ascii_characters_not_encoded_as_unicode():
    with mlflow.start_span() as span:
        span.set_inputs({"japanese": "あ", "emoji": "👍"})

    trace = mlflow.get_trace(span.trace_id)
    span = trace.data.spans[0]
    assert span.inputs == {"japanese": "あ", "emoji": "👍"}

    artifact_location = local_file_uri_to_path(trace.info.tags["mlflow.artifactLocation"])
    data = Path(artifact_location, "traces.json").read_text()
    assert "あ" in data
    assert "👍" in data
    assert json.dumps("あ").strip('"') not in data
    assert json.dumps("👍").strip('"') not in data


_SAMPLE_REMOTE_TRACE = {
    "info": {
        "request_id": "2e72d64369624e6888324462b62dc120",
        "experiment_id": "0",
        "timestamp_ms": 1726145090860,
        "execution_time_ms": 162,
        "status": "OK",
        "request_metadata": {
            "mlflow.trace_schema.version": "2",
            "mlflow.traceInputs": '{"x": 1}',
            "mlflow.traceOutputs": '{"prediction": 1}',
        },
        "tags": {
            "fruit": "apple",
            "food": "pizza",
        },
    },
    "data": {
        "spans": [
            {
                "name": "remote",
                "context": {
                    "span_id": "0x337af925d6629c01",
                    "trace_id": "0x05e82d1fc4486f3986fae6dd7b5352b1",
                },
                "parent_id": None,
                "start_time": 1726145091022155863,
                "end_time": 1726145091022572053,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"2e72d64369624e6888324462b62dc120"',
                    "mlflow.spanType": '"UNKNOWN"',
                    "mlflow.spanInputs": '{"x": 1}',
                    "mlflow.spanOutputs": '{"prediction": 1}',
                },
                "events": [
                    {"name": "event", "timestamp": 1726145091022287, "attributes": {"foo": "bar"}}
                ],
            },
            {
                "name": "remote-child",
                "context": {
                    "span_id": "0xa3dde9f2ebac1936",
                    "trace_id": "0x05e82d1fc4486f3986fae6dd7b5352b1",
                },
                "parent_id": "0x337af925d6629c01",
                "start_time": 1726145091022419340,
                "end_time": 1726145091022497944,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": '"2e72d64369624e6888324462b62dc120"',
                    "mlflow.spanType": '"UNKNOWN"',
                },
                "events": [],
            },
        ],
        "request": '{"x": 1}',
        "response": '{"prediction": 1}',
    },
}


def test_add_trace():
    # Mimic a remote service call that returns a trace as a part of the response
    def dummy_remote_call():
        return {"prediction": 1, "trace": _SAMPLE_REMOTE_TRACE}

    @mlflow.trace
    def predict(add_trace: bool):
        resp = dummy_remote_call()

        if add_trace:
            mlflow.add_trace(resp["trace"])
        return resp["prediction"]

    # If we don't call add_trace, the trace from the remote service should be discarded
    predict(add_trace=False)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1

    # If we call add_trace, the trace from the remote service should be merged
    predict(add_trace=True)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_id = trace.info.trace_id
    assert trace_id is not None
    assert trace.data.request == '{"add_trace": true}'
    assert trace.data.response == "1"
    # Remote spans should be merged
    assert len(trace.data.spans) == 3
    assert all(span.trace_id == trace_id for span in trace.data.spans)
    parent_span, child_span, grandchild_span = trace.data.spans
    assert child_span.parent_id == parent_span.span_id
    assert child_span._trace_id == parent_span._trace_id
    assert grandchild_span.parent_id == child_span.span_id
    assert grandchild_span._trace_id == parent_span._trace_id
    # Check if span information is correctly copied
    rs = Trace.from_dict(_SAMPLE_REMOTE_TRACE).data.spans[0]
    assert child_span.name == rs.name
    assert child_span.start_time_ns == rs.start_time_ns
    assert child_span.end_time_ns == rs.end_time_ns
    assert child_span.status == rs.status
    assert child_span.span_type == rs.span_type
    assert child_span.events == rs.events
    # exclude request ID attribute from comparison
    for k in rs.attributes.keys() - {SpanAttributeKey.REQUEST_ID}:
        assert child_span.attributes[k] == rs.attributes[k]


def test_add_trace_no_current_active_trace():
    # Use the remote trace without any active trace
    remote_trace = Trace.from_dict(_SAMPLE_REMOTE_TRACE)

    mlflow.add_trace(remote_trace)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 3
    parent_span, child_span, grandchild_span = trace.data.spans
    assert parent_span.name == "Remote Trace <remote>"
    rs = remote_trace.data.spans[0]
    assert parent_span.start_time_ns == rs.start_time_ns
    assert parent_span.end_time_ns == rs.end_time_ns
    assert child_span.name == rs.name
    assert child_span.parent_id is parent_span.span_id
    assert child_span.start_time_ns == rs.start_time_ns
    assert child_span.end_time_ns == rs.end_time_ns
    assert child_span.status == rs.status
    assert child_span.span_type == rs.span_type
    assert child_span.events == rs.events
    assert grandchild_span.parent_id == child_span.span_id
    # exclude request ID attribute from comparison
    for k in rs.attributes.keys() - {SpanAttributeKey.REQUEST_ID}:
        assert child_span.attributes[k] == rs.attributes[k]


def test_add_trace_specific_target_span():
    span = start_span_no_context(name="parent")
    mlflow.add_trace(_SAMPLE_REMOTE_TRACE, target=span)
    span.end()

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 3
    parent_span, child_span, grandchild_span = trace.data.spans
    assert parent_span.span_id == span.span_id
    rs = Trace.from_dict(_SAMPLE_REMOTE_TRACE).data.spans[0]
    assert child_span.name == rs.name
    assert child_span.parent_id is parent_span.span_id
    assert grandchild_span.parent_id == child_span.span_id


def test_add_trace_merge_tags():
    client = TracingClient()

    # Start the parent trace and merge the above trace as a child
    with mlflow.start_span(name="parent") as span:
        client.set_trace_tag(span.trace_id, "vegetable", "carrot")
        client.set_trace_tag(span.trace_id, "food", "sushi")

        mlflow.add_trace(Trace.from_dict(_SAMPLE_REMOTE_TRACE))

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    custom_tags = {k: v for k, v in trace.info.tags.items() if not k.startswith("mlflow.")}
    assert custom_tags == {
        "fruit": "apple",
        "vegetable": "carrot",
        # Tag value from the parent trace should prevail
        "food": "sushi",
    }


def test_add_trace_raise_for_invalid_trace():
    with pytest.raises(MlflowException, match="Invalid trace object"):
        mlflow.add_trace(None)

    with pytest.raises(MlflowException, match="Failed to load a trace object"):
        mlflow.add_trace({"info": {}, "data": {}})

    in_progress_trace = Trace(
        info=TraceInfo(
            trace_id="123",
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=0,
            execution_duration=0,
            state=TraceState.IN_PROGRESS,
        ),
        data=TraceData(),
    )
    with pytest.raises(MlflowException, match="The trace must be ended"):
        mlflow.add_trace(in_progress_trace)

    trace = Trace.from_dict(_SAMPLE_REMOTE_TRACE)
    spans = trace.data.spans
    unordered_trace = Trace(info=trace.info, data=TraceData(spans=[spans[1], spans[0]]))
    with pytest.raises(MlflowException, match="Span with ID "):
        mlflow.add_trace(unordered_trace)


@skip_when_testing_trace_sdk
def test_add_trace_in_databricks_model_serving(mock_databricks_serving_with_tracing_env):
    from mlflow.pyfunc.context import Context, set_prediction_context

    # Mimic a remote service call that returns a trace as a part of the response
    def dummy_remote_call():
        return {"prediction": 1, "trace": _SAMPLE_REMOTE_TRACE}

    # The parent function that invokes the dummy remote service
    @mlflow.trace
    def predict():
        resp = dummy_remote_call()
        remote_trace = Trace.from_dict(resp["trace"])
        mlflow.add_trace(remote_trace)
        return resp["prediction"]

    db_request_id = "databricks-request-id"
    with set_prediction_context(Context(request_id=db_request_id)):
        predict()

    # Pop the trace to be written to the inference table
    trace = Trace.from_dict(pop_trace(request_id=db_request_id))

    assert trace.info.trace_id.startswith("tr-")
    assert trace.info.client_request_id == db_request_id
    assert len(trace.data.spans) == 3
    assert all(span.trace_id == trace.info.trace_id for span in trace.data.spans)
    parent_span, child_span, grandchild_span = trace.data.spans
    assert child_span.parent_id == parent_span.span_id
    assert child_span._trace_id == parent_span._trace_id
    assert grandchild_span.parent_id == child_span.span_id
    assert grandchild_span._trace_id == parent_span._trace_id
    # Check if span information is correctly copied
    rs = Trace.from_dict(_SAMPLE_REMOTE_TRACE).data.spans[0]
    assert child_span.name == rs.name
    assert child_span.start_time_ns == rs.start_time_ns
    assert child_span.end_time_ns == rs.end_time_ns


@skip_when_testing_trace_sdk
def test_add_trace_logging_model_from_code():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model="tests/tracing/sample_code/model_with_add_trace.py",
            input_example=[1, 2],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    # Trace should not be logged while logging / loading
    assert mlflow.get_trace(mlflow.get_last_active_trace_id()) is None

    loaded_model.predict(1)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert len(trace.data.spans) == 2


@pytest.mark.parametrize(
    "inputs", [{"question": "Does mlflow support tracing?"}, "Does mlflow support tracing?", None]
)
@pytest.mark.parametrize("outputs", [{"answer": "Yes"}, "Yes", None])
@pytest.mark.parametrize(
    "intermediate_outputs",
    [
        {
            "retrieved_documents": ["mlflow documentation"],
            "system_prompt": ["answer the question with yes or no"],
        },
        None,
    ],
)
def test_log_trace_success(inputs, outputs, intermediate_outputs):
    start_time_ms = 1736144700
    execution_time_ms = 5129

    mlflow.log_trace(
        name="test",
        request=inputs,
        response=outputs,
        intermediate_outputs=intermediate_outputs,
        start_time_ms=start_time_ms,
        execution_time_ms=execution_time_ms,
    )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    if inputs is not None:
        assert trace.data.request == json.dumps(inputs)
    else:
        assert trace.data.request is None
    if outputs is not None:
        assert trace.data.response == json.dumps(outputs)
    else:
        assert trace.data.response is None
    if intermediate_outputs is not None:
        assert trace.data.intermediate_outputs == intermediate_outputs
    spans = trace.data.spans
    assert len(spans) == 1
    root_span = spans[0]
    assert root_span.name == "test"
    assert root_span.start_time_ns == start_time_ms * 1000000
    assert root_span.end_time_ns == (start_time_ms + execution_time_ms) * 1000000


def test_set_delete_trace_tag():
    with mlflow.start_span("span1") as span:
        trace_id = span.trace_id

    mlflow.set_trace_tag(trace_id=trace_id, key="key1", value="value1")
    trace = mlflow.get_trace(trace_id=trace_id)
    assert trace.info.tags["key1"] == "value1"

    mlflow.delete_trace_tag(trace_id=trace_id, key="key1")
    trace = mlflow.get_trace(trace_id=trace_id)
    assert "key1" not in trace.info.tags

    # Test with request_id kwarg (backward compatibility)
    mlflow.set_trace_tag(request_id=trace_id, key="key3", value="value3")
    trace = mlflow.get_trace(request_id=trace_id)
    assert trace.info.tags["key3"] == "value3"

    mlflow.delete_trace_tag(request_id=trace_id, key="key3")
    trace = mlflow.get_trace(request_id=trace_id)
    assert "key3" not in trace.info.tags


@pytest.mark.parametrize("is_databricks", [True, False])
def test_search_traces_with_run_id_validates_store_filter_string(is_databricks):
    mock_store = mock.MagicMock()
    mock_store.search_traces.return_value = ([], None)
    mock_store.get_run.return_value = mock.MagicMock()
    mock_store.get_run.return_value.info.experiment_id = "test_exp_id"

    test_run_id = "test_run_123"
    with (
        mock.patch("mlflow.tracing.client._get_store", return_value=mock_store),
        mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="test_exp_id"),
    ):
        mlflow.search_traces(run_id=test_run_id)

        expected_filter_string = f"attribute.run_id = '{test_run_id}'"
        mock_store.search_traces.assert_called()

        call_args = mock_store.search_traces.call_args
        actual_filter_string = call_args[1]["filter_string"]
        assert actual_filter_string == expected_filter_string


@skip_when_testing_trace_sdk
def test_set_destination_in_threads(async_logging_enabled):
    # This test makes sure `set_destination` obeys thread-local behavior.
    class TestModel:
        def predict(self, x):
            with mlflow.start_span(name="root_span") as root_span:

                def child_span_thread(z):
                    child_span = start_span_no_context(
                        name="child_span_1",
                        parent_span=root_span,
                    )
                    child_span.set_inputs(z)
                    time.sleep(0.5)
                    child_span.end()

                thread = threading.Thread(target=child_span_thread, args=(x + 1,))
                thread.start()
                thread.join()
            return x

    model = TestModel()

    def func(experiment_id: str | None, x: int):
        if experiment_id is not None:
            set_destination(MlflowExperiment(experiment_id), context_local=True)

        time.sleep(0.5)
        model.predict(x)

    # Main thread: global config
    experiment_id1 = mlflow.create_experiment(uuid.uuid4().hex)
    set_destination(MlflowExperiment(experiment_id1))
    func(None, 3)

    # Thread 1: context-local config
    experiment_id2 = mlflow.create_experiment(uuid.uuid4().hex)
    thread1 = threading.Thread(target=func, args=(experiment_id2, 3))

    # Thread 2: context-local config
    experiment_id3 = mlflow.create_experiment(uuid.uuid4().hex)
    thread2 = threading.Thread(target=func, args=(experiment_id3, 40))

    # Thread 3: no config -> fallback to global config
    thread3 = threading.Thread(target=func, args=(None, 40))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = get_traces(experiment_id1)
    assert len(traces) == 2  # main thread + thread 3
    assert traces[0].info.experiment_id == experiment_id1
    assert len(traces[0].data.spans) == 2
    assert traces[1].info.experiment_id == experiment_id1
    assert len(traces[1].data.spans) == 2

    for exp_id in [experiment_id2, experiment_id3]:
        traces = get_traces(exp_id)
        assert len(traces) == 1
        assert traces[0].info.experiment_id == exp_id
        assert len(traces[0].data.spans) == 2


@pytest.mark.asyncio
@skip_when_testing_trace_sdk
async def test_set_destination_in_async_contexts(async_logging_enabled):
    class TestModel:
        async def predict(self, x):
            with mlflow.start_span(name="root_span") as root_span:

                async def child_span_task(z):
                    child_span = start_span_no_context(
                        name="child_span_1",
                        parent_span=root_span,
                    )
                    child_span.set_inputs(z)
                    await asyncio.sleep(0.5)
                    child_span.end()

                await child_span_task(x + 1)
            return x

    model = TestModel()

    async def async_func(experiment_id: str, x: int):
        set_destination(MlflowExperiment(experiment_id), context_local=True)
        await asyncio.sleep(0.5)
        await model.predict(x)

    experiment_id1 = mlflow.create_experiment(uuid.uuid4().hex)
    task1 = asyncio.create_task(async_func(experiment_id1, 3))

    experiment_id2 = mlflow.create_experiment(uuid.uuid4().hex)
    task2 = asyncio.create_task(async_func(experiment_id2, 40))

    await asyncio.gather(task1, task2)

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    for exp_id in [experiment_id1, experiment_id2]:
        traces = get_traces(exp_id)
        assert len(traces) == 1
        assert traces[0].info.experiment_id == exp_id
        assert len(traces[0].data.spans) == 2


@skip_when_testing_trace_sdk
def test_traces_can_be_searched_by_span_properties(async_logging_enabled):
    """Smoke test that traces can be searched by span name using filter_string."""

    @mlflow.trace(name="test_span")
    def test_function():
        return "result"

    test_function()

    if async_logging_enabled:
        mlflow.flush_trace_async_logging(terminate=True)

    traces = mlflow.search_traces(filter_string='span.name = "test_span"', return_type="list")
    assert len(traces) == 1, "Should find exactly one trace with span name 'test_span'"
    found_span_names = [span.name for span in traces[0].data.spans]
    assert "test_span" in found_span_names
