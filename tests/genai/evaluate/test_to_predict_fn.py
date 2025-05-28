import json
from unittest import mock

import pytest

import mlflow
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
from mlflow.genai.evaluation.base import to_predict_fn
from mlflow.genai.utils.trace_utils import convert_predict_fn

from tests.evaluate.test_evaluation import _DUMMY_CHAT_RESPONSE


@pytest.fixture
def mock_deploy_client():
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get:
        yield mock_get.return_value


# TODO: Remove this once OSS backend is migrated to V3.
@pytest.fixture
def mock_tracing_client(monkeypatch):
    # Mock the TracingClient
    with mock.patch("mlflow.tracing.export.mlflow_v3.TracingClient") as mock_get:
        # Set up trace exporter to Databricks.
        monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, "false")
        mlflow.set_tracking_uri("databricks")
        mlflow.tracing.enable()  # Set up trace exporter again

        yield mock_get.return_value


def test_to_predict_fn_return_trace(sample_rag_trace, mock_deploy_client, mock_tracing_client):
    mock_deploy_client.predict.return_value = {
        **_DUMMY_CHAT_RESPONSE,
        "databricks_output": {"trace": sample_rag_trace.to_dict()},
    }
    messages = [
        {"content": "You are a helpful assistant.", "role": "system"},
        {"content": "What is Spark?", "role": "user"},
    ]

    predict_fn = to_predict_fn("endpoints:/chat")
    response = predict_fn(messages=messages)

    mock_deploy_client.predict.assert_called_once_with(
        endpoint="chat",
        inputs={
            "messages": messages,
            "databricks_options": {"return_trace": True},
        },
    )
    assert response == _DUMMY_CHAT_RESPONSE  # Response should not contain databricks_output

    # Trace from endpoint (sample_rag_trace) should be copied to the current experiment
    mock_tracing_client.start_trace_v3.assert_called_once()
    trace = mock_tracing_client.start_trace_v3.call_args[0][0]
    # Copied trace should have a new trace ID
    assert trace.info.trace_id != sample_rag_trace.info.trace_id
    assert trace.info.request_preview == '{"question": "query"}'
    assert trace.info.response_preview == '"answer"'
    assert len(trace.data.spans) == 3
    assert trace.data.spans[0].name == "rag"
    assert trace.data.spans[0].inputs == {"question": "query"}
    assert trace.data.spans[0].outputs == "answer"
    mock_tracing_client._upload_trace_data.assert_called_once_with(mock.ANY, trace.data)


@pytest.mark.parametrize(
    "databricks_output",
    [
        {},
        {"databricks_output": {}},
        {"databricks_output": {"trace": None}},
    ],
)
def test_to_predict_fn_does_not_return_trace(
    databricks_output, mock_deploy_client, mock_tracing_client
):
    mock_deploy_client.predict.return_value = {**_DUMMY_CHAT_RESPONSE, **databricks_output}
    messages = [
        {"content": "You are a helpful assistant.", "role": "system"},
        {"content": "What is Spark?", "role": "user"},
    ]

    predict_fn = to_predict_fn("endpoints:/chat")
    response = predict_fn(messages=messages)

    mock_deploy_client.predict.assert_called_once_with(
        endpoint="chat",
        inputs={
            "messages": messages,
            "databricks_options": {"return_trace": True},
        },
    )
    assert response == _DUMMY_CHAT_RESPONSE  # Response should not contain databricks_output

    # Bare-minimum trace should be created when the endpoint does not return a trace
    mock_tracing_client.start_trace_v3.assert_called_once()
    trace = mock_tracing_client.start_trace_v3.call_args[0][0]
    assert trace.info.request_preview == json.dumps({"messages": messages})
    assert len(trace.data.spans) == 1
    assert trace.data.spans[0].name == "predict"


def test_to_predict_fn_pass_tracing_check(
    sample_rag_trace, mock_deploy_client, mock_tracing_client
):
    """
    The function produced by to_predict_fn() is guaranteed to create a trace.
    Therefore it should not be wrapped by @mlflow.trace by convert_predict_fn().
    """
    mock_deploy_client.predict.return_value = {
        **_DUMMY_CHAT_RESPONSE,
        "databricks_output": {"trace": sample_rag_trace.to_dict()},
    }
    sample_input = {"messages": [{"role": "user", "content": "Hi"}]}

    predict_fn = to_predict_fn("endpoints:/chat")
    converted = convert_predict_fn(predict_fn, sample_input)

    # The check should pass, the function should not be wrapped by @mlflow.trace
    wrapped = hasattr(converted, "__wrapped__")
    assert wrapped != predict_fn

    # The function should not produce a trace during the check
    mock_tracing_client.start_trace_v3.assert_not_called()

    # The function should produce a trace when invoked
    converted(sample_input)

    mock_tracing_client.start_trace_v3.assert_called_once()
    trace = mock_tracing_client.start_trace_v3.call_args[0][0]
    assert trace.info.request_preview == '{"question": "query"}'
    assert trace.info.response_preview == '"answer"'
    # The produced trace should be the one returned from the endpoint (sample_rag_trace)
    assert trace.data.spans[0].name == "rag"
    assert trace.data.spans[0].inputs == {"question": "query"}
    assert trace.data.spans[0].outputs == "answer"
