import importlib
import json

import pytest
from packaging.version import Version

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.trace import _TraceJSONEncoder

from tests.tracing.conftest import mock_client as mock_trace_client  # noqa: F401


def test_json_deserialization(mock_trace_client):
    class TestModel:
        @mlflow.trace()
        def predict(self, x, y):
            z = x + y
            z = self.add_one(z)
            return z  # noqa: RET504

        @mlflow.trace(
            span_type=SpanType.LLM, name="add_one_with_custom_name", attributes={"delta": 1}
        )
        def add_one(self, z):
            return z + 1

    model = TestModel()
    model.predict(2, 5)

    trace = mlflow.get_traces()[0]
    trace_json = trace.to_json()

    trace_json_as_dict = json.loads(trace_json)

    assert trace_json_as_dict == {
        "trace_info": {
            "request_id": trace.trace_info.request_id,
            "experiment_id": "EXPERIMENT",
            "timestamp_ms": trace.trace_info.timestamp_ms,
            "execution_time_ms": trace.trace_info.execution_time_ms,
            "status": "OK",
            "request_metadata": {
                "name": "predict",
                "inputs": '{"x": 2, "y": 5}',
                "outputs": '{"output": 8}',
            },
            "tags": {},
        },
        "trace_data": {
            "spans": [
                {
                    "name": "predict",
                    "context": {
                        "request_id": trace.trace_data.spans[0].context.request_id,
                        "span_id": trace.trace_data.spans[0].context.span_id,
                    },
                    "span_type": "UNKNOWN",
                    "parent_span_id": None,
                    "start_time": trace.trace_data.spans[0].start_time,
                    "end_time": trace.trace_data.spans[0].end_time,
                    "status": {"status_code": "OK", "description": ""},
                    "inputs": {"x": 2, "y": 5},
                    "outputs": {"output": 8},
                    "attributes": {"function_name": "predict"},
                    "events": [],
                },
                {
                    "name": "add_one_with_custom_name",
                    "context": {
                        "request_id": trace.trace_data.spans[1].context.request_id,
                        "span_id": trace.trace_data.spans[1].context.span_id,
                    },
                    "span_type": "LLM",
                    "parent_span_id": trace.trace_data.spans[0].context.span_id,
                    "start_time": trace.trace_data.spans[1].start_time,
                    "end_time": trace.trace_data.spans[1].end_time,
                    "status": {"status_code": "OK", "description": ""},
                    "inputs": {"z": 7},
                    "outputs": {"output": 8},
                    "attributes": {"delta": 1, "function_name": "add_one"},
                    "events": [],
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
    data_json = json.dumps(data, cls=_TraceJSONEncoder)
    assert data_json == '{"x": 1, "y": "foo"}'
    assert json.loads(data_json) == {"x": 1, "y": "foo"}


def _is_langchain_v0_1():
    try:
        import langchain

        return Version(langchain.__version__) >= Version("0.1")
    except ImportError:
        return None


@pytest.mark.skipif(not _is_langchain_v0_1(), reason="langchain is not installed")
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

    message_json = json.dumps(message, cls=_TraceJSONEncoder)
    # "additional_kwargs" field is added by Langchain
    assert message_json == (
        '{"content": [{"role": "system", "content": "Hello, World!"}, '
        '{"role": "user", "content": "Hi!"}], "additional_kwargs": {}, "type": "chat"}'
    )
    assert json.loads(message_json) == {
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
        "additional_kwargs": {},
    }
