import importlib
import json
import re
from datetime import datetime
from unittest import mock

import pytest
from packaging.version import Version

import mlflow
import mlflow.tracking.context.default_context
from mlflow.entities import SpanType, Trace, TraceData
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.utils.proto_json_utils import (
    milliseconds_to_proto_timestamp,
)

from tests.tracing.helper import create_test_trace_info


def _test_model(datetime=datetime.now()):
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            return z  # noqa: RET504

        @mlflow.trace(
            span_type=SpanType.LLM,
            name="add_one_with_custom_name",
            attributes={
                "delta": 1,
                "metadata": {"foo": "bar"},
                # Test for non-json-serializable input
                "datetime": datetime,
            },
        )
        def add_one(self, z):
            return z + 1

    return TestModel()


def test_json_deserialization(monkeypatch):
    monkeypatch.setattr(mlflow.tracking.context.default_context, "_get_source_name", lambda: "test")
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "bob")
    datetime_now = datetime.now()

    model = _test_model(datetime_now)
    model.predict(2, 5)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    trace_json = trace.to_json()

    trace_json_as_dict = json.loads(trace_json)
    assert trace_json_as_dict == {
        "info": {
            "trace_id": trace.info.request_id,
            "trace_location": {
                "mlflow_experiment": {
                    "experiment_id": "0",
                },
                "type": "MLFLOW_EXPERIMENT",
            },
            "request_time": milliseconds_to_proto_timestamp(trace.info.timestamp_ms),
            "execution_duration_ms": trace.info.execution_time_ms,
            "state": "OK",
            "request_preview": '{"x": 2, "y": 5}',
            "response_preview": "8",
            "trace_metadata": {
                TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
                "mlflow.traceInputs": '{"x": 2, "y": 5}',
                "mlflow.traceOutputs": "8",
                "mlflow.source.name": mock.ANY,
                "mlflow.source.type": "LOCAL",
                "mlflow.source.git.branch": mock.ANY,
                "mlflow.source.git.commit": mock.ANY,
                "mlflow.source.git.repoURL": mock.ANY,
                "mlflow.user": mock.ANY,
            },
            "tags": {
                "mlflow.traceName": "predict",
                "mlflow.artifactLocation": trace.info.tags[MLFLOW_ARTIFACT_LOCATION],
            },
        },
        "data": {
            "spans": [
                {
                    "name": "predict",
                    "trace_id": mock.ANY,
                    "span_id": mock.ANY,
                    "parent_span_id": "",
                    "start_time_unix_nano": trace.data.spans[0].start_time_ns,
                    "end_time_unix_nano": trace.data.spans[0].end_time_ns,
                    "status": {
                        "code": "STATUS_CODE_OK",
                        "message": "",
                    },
                    "trace_state": "",
                    "attributes": {
                        "mlflow.traceRequestId": json.dumps(trace.info.request_id),
                        "mlflow.spanType": '"UNKNOWN"',
                        "mlflow.spanFunctionName": '"predict"',
                        "mlflow.spanInputs": '{"x": 2, "y": 5}',
                        "mlflow.spanOutputs": "8",
                    },
                },
                {
                    "name": "add_one_with_custom_name",
                    "trace_id": mock.ANY,
                    "span_id": mock.ANY,
                    "parent_span_id": mock.ANY,
                    "start_time_unix_nano": trace.data.spans[1].start_time_ns,
                    "end_time_unix_nano": trace.data.spans[1].end_time_ns,
                    "status": {
                        "code": "STATUS_CODE_OK",
                        "message": "",
                    },
                    "trace_state": "",
                    "attributes": {
                        "mlflow.traceRequestId": json.dumps(trace.info.request_id),
                        "mlflow.spanType": '"LLM"',
                        "mlflow.spanFunctionName": '"add_one"',
                        "mlflow.spanInputs": '{"z": 7}',
                        "mlflow.spanOutputs": "8",
                        "delta": "1",
                        "datetime": json.dumps(str(datetime_now)),
                        "metadata": '{"foo": "bar"}',
                    },
                },
            ],
        },
    }


@pytest.mark.skipif(
    importlib.util.find_spec("pydantic") is None, reason="Pydantic is not installed"
)
def test_trace_serialize_pydantic_model():
    from pydantic import BaseModel

    class MyModel(BaseModel):
        x: int
        y: str

    data = MyModel(x=1, y="foo")
    data_json = json.dumps(data, cls=TraceJSONEncoder)
    assert data_json == '{"x": 1, "y": "foo"}'
    assert json.loads(data_json) == {"x": 1, "y": "foo"}


def _is_langchain_v0_1():
    try:
        import langchain

        return Version(langchain.__version__) >= Version("0.1")
    except ImportError:
        return None


@pytest.mark.skipif(not _is_langchain_v0_1(), reason="langchain>=0.1 is not installed")
def test_trace_serialize_langchain_base_message():
    from langchain_core.messages import BaseMessage

    message = BaseMessage(
        content=[
            {
                "role": "system",
                "content": "Hello, World!",
            },
            {
                "role": "user",
                "content": "Hi!",
            },
        ],
        type="chat",
    )

    message_json = json.dumps(message, cls=TraceJSONEncoder)
    # LangChain message model contains a few more default fields actually. But we
    # only check if the following subset of the expected dictionary is present in
    # the loaded JSON rather than exact equality, because the LangChain BaseModel
    # has been changing frequently and the additional default fields may differ
    # across versions installed on developers' machines.
    expected_dict_subset = {
        "content": [
            {
                "role": "system",
                "content": "Hello, World!",
            },
            {
                "role": "user",
                "content": "Hi!",
            },
        ],
        "type": "chat",
    }
    loaded = json.loads(message_json)
    assert expected_dict_subset.items() <= loaded.items()


def test_trace_to_from_dict_and_json():
    model = _test_model()
    model.predict(2, 5)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    spans = trace.search_spans(span_type=SpanType.LLM)
    assert len(spans) == 1

    spans = trace.search_spans(name="predict")
    assert len(spans) == 1

    trace_dict = trace.to_dict()
    trace_from_dict = Trace.from_dict(trace_dict)
    trace_json = trace.to_json()
    trace_from_json = Trace.from_json(trace_json)
    for loaded_trace in [trace_from_dict, trace_from_json]:
        assert trace.info == loaded_trace.info
        assert trace.data.request == loaded_trace.data.request
        assert trace.data.response == loaded_trace.data.response
        assert len(trace.data.spans) == len(loaded_trace.data.spans)
        for i in range(len(trace.data.spans)):
            for attr in [
                "name",
                "request_id",
                "span_id",
                "start_time_ns",
                "end_time_ns",
                "parent_id",
                "status",
                "inputs",
                "outputs",
                "_trace_id",
                "attributes",
                "events",
            ]:
                assert getattr(trace.data.spans[i], attr) == getattr(
                    loaded_trace.data.spans[i], attr
                )


def test_trace_pandas_dataframe_columns():
    t = Trace(
        info=create_test_trace_info("a"),
        data=TraceData(),
    )
    assert Trace.pandas_dataframe_columns() == list(t.to_pandas_dataframe_row())


@pytest.mark.parametrize(
    ("span_type", "name", "expected"),
    [
        (None, None, ["run", "add_one_1", "add_one_2", "add_two", "multiply_by_two"]),
        (SpanType.CHAIN, None, ["run"]),
        (None, "add_two", ["add_two"]),
        (None, re.compile(r"add.*"), ["add_one_1", "add_one_2", "add_two"]),
        (None, re.compile(r"^add"), ["add_one_1", "add_one_2", "add_two"]),
        (None, re.compile(r"_two$"), ["add_two", "multiply_by_two"]),
        (None, re.compile(r".*ONE", re.IGNORECASE), ["add_one_1", "add_one_2"]),
        (SpanType.TOOL, "multiply_by_two", ["multiply_by_two"]),
        (SpanType.AGENT, None, []),
        (None, "non_existent", []),
    ],
)
def test_search_spans(span_type, name, expected):
    @mlflow.trace(span_type=SpanType.CHAIN)
    def run(x: int) -> int:
        x = add_one(x)
        x = add_one(x)
        x = add_two(x)
        return multiply_by_two(x)

    @mlflow.trace(span_type=SpanType.TOOL)
    def add_one(x: int) -> int:
        return x + 1

    @mlflow.trace(span_type=SpanType.TOOL)
    def add_two(x: int) -> int:
        return x + 2

    @mlflow.trace(span_type=SpanType.TOOL)
    def multiply_by_two(x: int) -> int:
        return x * 2

    run(2)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    spans = trace.search_spans(span_type=span_type, name=name)

    assert [span.name for span in spans] == expected


def test_search_spans_raise_for_invalid_param_type():
    @mlflow.trace(span_type=SpanType.CHAIN)
    def run(x: int) -> int:
        return x + 1

    run(2)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    with pytest.raises(MlflowException, match="Invalid type for 'span_type'"):
        trace.search_spans(span_type=123)

    with pytest.raises(MlflowException, match="Invalid type for 'name'"):
        trace.search_spans(name=123)


def test_from_v2_dict():
    v2_dict = {
        "info": {
            "request_id": "58f4e27101304034b15c512b603bf1b2",
            "experiment_id": "0",
            "timestamp_ms": 100,
            "execution_time_ms": 200,
            "status": "OK",
            "request_metadata": {
                "mlflow.trace_schema.version": "2",
                "mlflow.traceInputs": '{"x": 2, "y": 5}',
                "mlflow.traceOutputs": "8",
            },
            "tags": {
                "mlflow.source.name": "test",
                "mlflow.source.type": "LOCAL",
                "mlflow.traceName": "predict",
                "mlflow.artifactLocation": "/path/to/artifact",
            },
            "assessments": [],
        },
        "data": {
            "spans": [
                {
                    "name": "predict",
                    "context": {
                        "span_id": "0d48a6670588966b",
                        "trace_id": "63076d0c1b90f1df0970f897dc428bd6",
                    },
                    "parent_id": None,
                    "start_time": 100,
                    "end_time": 200,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"58f4e27101304034b15c512b603bf1b2"',
                        "mlflow.spanType": '"UNKNOWN"',
                        "mlflow.spanFunctionName": '"predict"',
                        "mlflow.spanInputs": '{"x": 2, "y": 5}',
                        "mlflow.spanOutputs": "8",
                    },
                    "events": [],
                },
                {
                    "name": "add_one_with_custom_name",
                    "context": {
                        "span_id": "6fc32f36ef591f60",
                        "trace_id": "63076d0c1b90f1df0970f897dc428bd6",
                    },
                    "parent_id": "0d48a6670588966b",
                    "start_time": 300,
                    "end_time": 400,
                    "status_code": "OK",
                    "status_message": "",
                    "attributes": {
                        "mlflow.traceRequestId": '"58f4e27101304034b15c512b603bf1b2"',
                        "mlflow.spanType": '"LLM"',
                        "delta": "1",
                        "metadata": '{"foo": "bar"}',
                        "datetime": '"2025-04-29 08:37:06.772253"',
                        "mlflow.spanFunctionName": '"add_one"',
                        "mlflow.spanInputs": '{"z": 7}',
                        "mlflow.spanOutputs": "8",
                    },
                    "events": [],
                },
            ],
            "request": '{"x": 2, "y": 5}',
            "response": "8",
        },
    }
    trace = Trace.from_dict(v2_dict)
    assert trace.info.request_id == "58f4e27101304034b15c512b603bf1b2"
    assert trace.info.request_time == 100
    assert trace.info.execution_duration == 200
    assert len(trace.data.spans) == 2
