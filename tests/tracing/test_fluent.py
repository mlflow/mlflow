import json
import time
from dataclasses import asdict
from datetime import datetime
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
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing.constant import (
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.provider import _get_tracer

from tests.tracing.helper import create_test_trace_info, create_trace, get_traces


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


@pytest.fixture
def mock_client():
    client = mock.MagicMock()
    with mock.patch("mlflow.tracing.fluent.MlflowClient", return_value=client):
        yield client


@pytest.mark.parametrize("with_active_run", [True, False])
def test_trace(with_active_run):
    model = DefaultTestModel()

    if with_active_run:
        with mlflow.start_run() as run:
            model.predict(2, 5)
            run_id = run.info.run_id
    else:
        model.predict(2, 5)

    trace = mlflow.get_last_active_trace()
    trace_info = trace.info
    assert trace_info.request_id is not None
    assert trace_info.experiment_id == "0"  # default experiment
    assert trace_info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace_info.status == SpanStatusCode.OK
    assert trace_info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace_info.request_metadata[TraceMetadataKey.OUTPUTS] == "64"
    if with_active_run:
        assert trace_info.request_metadata["mlflow.sourceRun"] == run_id

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response == "64"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["predict"]
    assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanFunctionName": "predict",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 2, "y": 5},
        "mlflow.spanOutputs": 64,
    }

    child_span_1 = span_name_to_span["add_one_with_custom_name"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 1,
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanFunctionName": "add_one",
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": {"z": 7},
        "mlflow.spanOutputs": 8,
    }

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace_info.request_id,
        "mlflow.spanFunctionName": "square",
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 8},
        "mlflow.spanOutputs": 64,
    }


def test_trace_with_databricks_tracking_uri(databricks_tracking_uri, mock_store, monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    mock_experiment = mock.MagicMock()
    mock_experiment.experiment_id = "test_experiment_id"
    monkeypatch.setattr(
        mock_store, "get_experiment_by_name", mock.MagicMock(return_value=mock_experiment)
    )

    model = DefaultTestModel()

    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data"
    ) as mock_upload_trace_data:
        model.predict(2, 5)

    traces = get_traces()
    assert len(traces) == 1
    trace_info = traces[0].info
    assert trace_info.request_id == "tr-12345"
    assert trace_info.experiment_id == "test_experiment_id"
    assert trace_info.status == TraceStatus.OK
    assert trace_info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace_info.request_metadata[TraceMetadataKey.OUTPUTS] == "64"
    assert trace_info.tags == {
        "mlflow.traceName": "predict",
        "mlflow.artifactLocation": "test",
        "mlflow.source.name": "test",
        "mlflow.source.type": "LOCAL",
        "mlflow.user": "bob",
        TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
    }

    trace_data = traces[0].data
    assert trace_data.request == '{"x": 2, "y": 5}'
    assert trace_data.response == "64"
    assert len(trace_data.spans) == 3

    mock_store.start_trace.assert_called_once()
    mock_store.end_trace.assert_called_once()
    mock_upload_trace_data.assert_called_once()


def test_trace_in_databricks_model_serving(mock_databricks_serving_with_tracing_env):
    # Dummy flask app for prediction
    import flask

    from mlflow.tracing.export.inference_table import pop_trace

    app = flask.Flask(__name__)

    @app.route("/invocations", methods=["POST"])
    def predict():
        data = json.loads(flask.request.data.decode("utf-8"))
        request_id = flask.request.headers.get("X-Request-ID")

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
    assert trace.info.request_id == databricks_request_id
    assert trace.info.tags[TRACE_SCHEMA_VERSION_KEY] == "2"
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
        "mlflow.traceRequestId": databricks_request_id,
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
        "mlflow.traceRequestId": databricks_request_id,
        "mlflow.spanType": SpanType.LLM,
        "mlflow.spanFunctionName": "add_one",
        "mlflow.spanInputs": {"z": 7},
        "mlflow.spanOutputs": 8,
    }
    assert child_span_1.events == []

    child_span_2 = span_name_to_span["square"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": databricks_request_id,
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


def test_trace_in_model_evaluation(mock_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")

    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return x + y

    model = TestModel()

    # mock _upload_trace_data to avoid generating trace data file
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data"
    ), mlflow.start_run() as run:
        run_id = run.info.run_id
        request_id_1 = "tr-eval-123"
        with set_prediction_context(Context(request_id=request_id_1, is_evaluate=True)):
            model.predict(1, 2)

        request_id_2 = "tr-eval-456"
        with set_prediction_context(Context(request_id=request_id_2, is_evaluate=True)):
            model.predict(3, 4)

    expected_tags = {
        "mlflow.traceName": "predict",
        "mlflow.source.name": "test",
        "mlflow.source.type": "LOCAL",
        "mlflow.user": "bob",
        "mlflow.artifactLocation": "test",
        TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
    }

    trace = mlflow.get_trace(request_id_1)
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
    assert trace.info.tags == {**expected_tags, **{TraceTagKey.EVAL_REQUEST_ID: request_id_1}}

    trace = mlflow.get_trace(request_id_2)
    assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
    assert trace.info.tags == {**expected_tags, **{TraceTagKey.EVAL_REQUEST_ID: request_id_2}}

    assert mock_store.start_trace.call_count == 2
    assert mock_store.end_trace.call_count == 2


def test_trace_handle_exception_during_prediction():
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

    with pytest.raises(ValueError, match=r"Some error"):
        model.predict(2, 5)

    # Trace should be logged even if the function fails, with status code ERROR
    trace = mlflow.get_last_active_trace()
    assert trace.info.request_id is not None
    assert trace.info.status == TraceStatus.ERROR
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 2, "y": 5}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == ""

    assert trace.data.request == '{"x": 2, "y": 5}'
    assert trace.data.response is None
    assert len(trace.data.spans) == 2


def test_trace_ignore_exception_from_tracing_logic(monkeypatch):
    # This test is to make sure that the main prediction logic is not affected
    # by the exception raised by the tracing logic.
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            return x + y

    model = TestModel()

    # Exception during span creation: no-op span wrapper created and no trace is logged
    with mock.patch("mlflow.tracing.provider._get_tracer", side_effect=ValueError("Some error")):
        output = model.predict(2, 5)

    assert output == 7
    assert get_traces() == []
    TRACE_BUFFER.clear()

    # Exception during inspecting inputs: trace is logged without inputs field
    with mock.patch(
        "mlflow.tracing.fluent.capture_function_input_args", side_effect=ValueError("Some error")
    ) as mock_input_args:
        output = model.predict(2, 5)
        mock_input_args.assert_called_once()

    assert output == 7
    trace = mlflow.get_last_active_trace()
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == ""
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "7"
    TRACE_BUFFER.clear()

    # Exception during ending span: trace is not logged
    # Mock the span processor's on_end handler to raise an exception
    tracer = _get_tracer(__name__)

    def _always_fail(*args, **kwargs):
        raise ValueError("Some error")

    monkeypatch.setattr(tracer.span_processor, "on_end", _always_fail)

    output = model.predict(2, 5)
    assert output == 7
    assert get_traces() == []
    TRACE_BUFFER.clear()


def test_start_span_context_manager():
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

    trace = mlflow.get_last_active_trace()
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == TraceStatus.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "25"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "25"
    assert len(trace.data.spans) == 3

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert (root_span.end_time_ns - root_span.start_time_ns) // 1e6 == trace.info.execution_time_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
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
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }

    child_span_2 = span_name_to_span["child_span_2"]
    assert child_span_2.parent_id == root_span.span_id
    assert child_span_2.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"t": 5},
        "mlflow.spanOutputs": 25,
    }
    assert child_span_2.start_time_ns <= child_span_2.end_time_ns - 0.1 * 1e6


def test_start_span_context_manager_with_imperative_apis():
    # This test is to make sure that the spans created with fluent APIs and imperative APIs
    # (via MLflow client) are correctly linked together. This usage is not recommended but
    # should be supported for the advanced use cases like using LangChain callbacks as a
    # part of broader tracing.
    class TestModel:
        def __init__(self):
            self._mlflow_client = mlflow.tracking.MlflowClient()

        def predict(self, x, y):
            with mlflow.start_span(name="root_span") as root_span:
                root_span.set_inputs({"x": x, "y": y})
                z = x + y

                child_span = self._mlflow_client.start_span(
                    name="child_span_1",
                    span_type=SpanType.LLM,
                    request_id=root_span.request_id,
                    parent_id=root_span.span_id,
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

    trace = mlflow.get_last_active_trace()
    assert trace.info.request_id is not None
    assert trace.info.experiment_id == "0"  # default experiment
    assert trace.info.execution_time_ms >= 0.1 * 1e3  # at least 0.1 sec
    assert trace.info.status == TraceStatus.OK
    assert trace.info.request_metadata[TraceMetadataKey.INPUTS] == '{"x": 1, "y": 2}'
    assert trace.info.request_metadata[TraceMetadataKey.OUTPUTS] == "5"

    assert trace.data.request == '{"x": 1, "y": 2}'
    assert trace.data.response == "5"
    assert len(trace.data.spans) == 2

    span_name_to_span = {span.name: span for span in trace.data.spans}
    root_span = span_name_to_span["root_span"]
    assert root_span.start_time_ns // 1e6 == trace.info.timestamp_ms
    assert (root_span.end_time_ns - root_span.start_time_ns) // 1e6 == trace.info.execution_time_ms
    assert root_span.parent_id is None
    assert root_span.attributes == {
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "UNKNOWN",
        "mlflow.spanInputs": {"x": 1, "y": 2},
        "mlflow.spanOutputs": 5,
    }

    child_span_1 = span_name_to_span["child_span_1"]
    assert child_span_1.parent_id == root_span.span_id
    assert child_span_1.attributes == {
        "delta": 2,
        "mlflow.traceRequestId": trace.info.request_id,
        "mlflow.spanType": "LLM",
        "mlflow.spanInputs": 3,
        "mlflow.spanOutputs": 5,
    }


def test_test_search_traces_empty(mock_client):
    mock_client.search_traces.return_value = PagedList([], token=None)

    traces = mlflow.search_traces()
    assert traces.empty

    default_columns = Trace.pandas_dataframe_columns()
    assert traces.columns.tolist() == default_columns

    traces = mlflow.search_traces(extract_fields=["foo.inputs.bar"])
    assert traces.columns.tolist() == [*default_columns, "foo.inputs.bar"]

    mock_client.search_traces.assert_called()


def test_search_traces(mock_client):
    mock_client.search_traces.return_value = PagedList(
        [
            Trace(
                info=create_test_trace_info(f"tr-{i}"),
                data=TraceData([], "", ""),
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
    )

    assert len(traces) == 10
    mock_client.search_traces.assert_called_once_with(
        experiment_ids=["1"],
        filter_string="name = 'foo'",
        max_results=10,
        order_by=["timestamp DESC"],
        page_token=None,
    )


def test_search_traces_with_pagination(mock_client):
    traces = [
        Trace(
            info=create_test_trace_info(f"tr-{i}"),
            data=TraceData([], "", ""),
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
        "max_results": SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        "filter_string": None,
        "order_by": None,
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
    with mock.patch("mlflow.tracing.fluent._get_experiment_id", return_value="123"):
        mlflow.search_traces()

    mock_client.search_traces.assert_called_once_with(
        experiment_ids=["123"],
        filter_string=None,
        max_results=SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by=None,
        page_token=None,
    )


def test_search_traces_yields_expected_dataframe_contents(monkeypatch):
    traces_to_return = [create_trace("a"), create_trace("b"), create_trace("c")]

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return traces_to_return

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces()
    assert df.columns.tolist() == [
        "request_id",
        "trace",
        "timestamp_ms",
        "status",
        "execution_time_ms",
        "request",
        "response",
        "request_metadata",
        "spans",
        "tags",
    ]
    for idx, trace in enumerate(traces_to_return):
        assert df.iloc[idx].request_id == trace.info.request_id
        assert df.iloc[idx].trace == trace
        assert df.iloc[idx].timestamp_ms == trace.info.timestamp_ms
        assert df.iloc[idx].status == trace.info.status
        assert df.iloc[idx].execution_time_ms == trace.info.execution_time_ms
        assert df.iloc[idx].request == trace.data.request
        assert df.iloc[idx].response == trace.data.response
        assert df.iloc[idx].request_metadata == trace.info.request_metadata
        assert df.iloc[idx].spans == trace.data.spans
        assert df.iloc[idx].tags == trace.info.tags


def test_search_traces_handles_missing_response_tags_and_metadata(monkeypatch):
    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return [
                Trace(
                    info=TraceInfo(
                        request_id=5,
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=2,
                        status=TraceStatus.OK,
                    ),
                    data=TraceData(
                        spans=[],
                        request="request",
                        # Response is missing
                    ),
                )
            ]

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces()
    assert df["response"].isnull().all()
    assert df["tags"].tolist() == [{}]
    assert df["request_metadata"].tolist() == [{}]


def test_search_traces_extracts_fields_as_expected(monkeypatch):
    model = DefaultTestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=["predict.inputs.x", "predict.outputs", "add_one_with_custom_name.inputs.z"]
    )
    assert df["predict.inputs.x"].tolist() == [2]
    assert df["predict.outputs"].tolist() == [64]
    assert df["add_one_with_custom_name.inputs.z"].tolist() == [7]


# Test cases should cover case where there are no spans at all
def test_search_traces_with_no_spans(monkeypatch):
    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return []

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces()
    assert df.empty


# no spans have the input or output with name,
# some span has an input but we’re looking for output,
def test_search_traces_with_input_and_no_output(monkeypatch):
    with mlflow.start_span(name="with_input_and_no_output") as span:
        span.set_inputs({"a": 1})

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=["with_input_and_no_output.inputs.a", "with_input_and_no_output.outputs"]
    )
    assert df["with_input_and_no_output.inputs.a"].tolist() == [1]
    assert df["with_input_and_no_output.outputs"].isnull().all()


# Test case where span content is invalid
def test_search_traces_with_invalid_span_content(monkeypatch):
    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            # Invalid span content
            return [
                Trace(
                    info=TraceInfo(
                        request_id=5,
                        experiment_id="test",
                        timestamp_ms=1,
                        execution_time_ms=2,
                        status=TraceStatus.OK,
                    ),
                    data=TraceData(spans=[None], request="request", response="response"),
                )
            ]

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    with pytest.raises(AttributeError, match="NoneType"):
        mlflow.search_traces()


# Test case where span inputs / outputs aren’t dict
def test_search_traces_with_non_dict_span_inputs_outputs(monkeypatch):
    with mlflow.start_span(name="non_dict_span") as span:
        span.set_inputs(["a", "b"])
        span.set_outputs([1, 2, 3])

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    df = mlflow.search_traces(
        extract_fields=["non_dict_span.inputs", "non_dict_span.outputs", "non_dict_span.inputs.x"]
    )
    assert df["non_dict_span.inputs"].tolist() == [["a", "b"]]
    assert df["non_dict_span.outputs"].tolist() == [[1, 2, 3]]
    assert df["non_dict_span.inputs.x"].isnull().all()


# Test case where there are multiple spans with the same name
def test_search_traces_with_multiple_spans_with_same_name(monkeypatch):
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

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

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


# Test a field that doesn’t exist for extraction - we shouldn’t throw, just return empty column
def test_search_traces_with_non_existent_field(monkeypatch):
    model = DefaultTestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

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


# Test experiment ID doesn’t need to be specified
def test_search_traces_without_experiment_id(monkeypatch):
    model = DefaultTestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, experiment_ids, *args, **kwargs):
            assert experiment_ids == ["0"]
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)

    mlflow.search_traces()


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


def test_search_traces_with_span_name(monkeypatch):
    class TestModel:
        @mlflow.trace(name="span.llm")
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            z = mlflow.trace(self.square)(z)
            return z  # noqa: RET504

        @mlflow.trace(span_type=SpanType.LLM, name="span.invalidname", attributes={"delta": 1})
        def add_one(self, z):
            return z + 1

        def square(self, t):
            res = t**2
            time.sleep(0.1)
            return res

    model = TestModel()
    model.predict(2, 5)

    class MockMlflowClient:
        def search_traces(self, experiment_ids, *args, **kwargs):
            return get_traces()

    monkeypatch.setattr("mlflow.tracing.fluent.MlflowClient", MockMlflowClient)


@pytest.mark.parametrize(
    "extract_fields",
    [
        ["span.llm.inputs"],
        ["span.llm.inputs.x"],
        ["span.llm.outputs"],
    ],
)
def test_search_traces_invalid_extract_fields(extract_fields):
    with pytest.raises(MlflowException, match="Invalid field type"):
        mlflow.search_traces(extract_fields=extract_fields)


def test_get_last_active_trace():
    assert mlflow.get_last_active_trace() is None

    @mlflow.trace()
    def predict(x, y):
        return x + y

    predict(1, 2)
    predict(2, 5)
    predict(3, 6)

    trace = mlflow.get_last_active_trace()
    assert trace.info.request_id is not None
    assert trace.data.request == '{"x": 3, "y": 6}'

    # Mutation of the copy should not affect the original trace logged in the backend
    trace.info.status = TraceStatus.ERROR
    original_trace = mlflow.MlflowClient().get_trace(trace.info.request_id)
    assert original_trace.info.status == TraceStatus.OK
