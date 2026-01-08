from unittest import mock

import pytest

import mlflow
from mlflow.entities.trace_info import TraceInfo
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
from mlflow.genai.evaluation.base import to_predict_fn
from mlflow.genai.utils.trace_utils import convert_predict_fn

from tests.tracing.helper import V2_TRACE_DICT

_DUMMY_CHAT_RESPONSE = {
    "id": "1",
    "object": "text_completion",
    "created": "2021-10-01T00:00:00.000000Z",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "content": "This is a response",
                "role": "assistant",
            },
            "finish_reason": "length",
        }
    ],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    },
}


@pytest.fixture
def mock_deploy_client():
    with mock.patch("mlflow.deployments.get_deploy_client") as mock_get:
        yield mock_get.return_value


# TODO: Remove this once OSS backend is migrated to V3.
@pytest.fixture
def mock_tracing_client(monkeypatch):
    # Mock the TracingClient
    with mock.patch("mlflow.tracing.export.mlflow_v3.TracingClient") as mock_get:
        tracing_client = mock_get.return_value
        tracing_client.tracking_uri = "databricks"

        # Set up trace exporter to Databricks.
        monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, "false")
        mlflow.set_tracking_uri("databricks")
        mlflow.tracing.enable()  # Set up trace exporter again

        yield tracing_client


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
    mock_tracing_client.start_trace.assert_called_once()
    trace_info = mock_tracing_client.start_trace.call_args[0][0]
    # Copied trace should have a new trace ID
    assert trace_info.trace_id != sample_rag_trace.info.trace_id
    assert trace_info.request_preview == '{"question": "query"}'
    assert trace_info.response_preview == '"answer"'

    trace_data = mock_tracing_client._upload_trace_data.call_args[0][1]
    assert len(trace_data.spans) == 3
    for old, new in zip(sample_rag_trace.data.spans, trace_data.spans):
        assert old.name == new.name
        assert old.inputs == new.inputs
        assert old.outputs == new.outputs
        assert old.start_time_ns == new.start_time_ns
        assert old.end_time_ns == new.end_time_ns
        assert old.parent_id == new.parent_id
        assert old.span_id == new.span_id
    mock_tracing_client._upload_trace_data.assert_called_once_with(mock.ANY, trace_data)


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
    mock_deploy_client.predict.return_value = _DUMMY_CHAT_RESPONSE | databricks_output
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
    mock_tracing_client.start_trace.assert_called_once()
    trace_info = mock_tracing_client.start_trace.call_args[0][0]
    assert trace_info.request_preview == "What is Spark?"
    trace_data = mock_tracing_client._upload_trace_data.call_args[0][1]
    assert len(trace_data.spans) == 1
    assert trace_data.spans[0].name == "predict"


def test_to_predict_fn_pass_tracing_check(
    sample_rag_trace, mock_deploy_client, mock_tracing_client
):
    """
    The function produced by to_predict_fn() is guaranteed to create a trace.
    Therefore it should not be wrapped by @mlflow.trace by convert_predict_fn().
    """
    mock_deploy_client.predict.side_effect = lambda **kwargs: {
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
    mock_tracing_client.start_trace.assert_not_called()

    # The function should produce a trace when invoked
    converted(sample_input)

    mock_tracing_client.start_trace.assert_called_once()
    trace_info = mock_tracing_client.start_trace.call_args[0][0]
    assert trace_info.request_preview == '{"question": "query"}'
    assert trace_info.response_preview == '"answer"'
    # The produced trace should be the one returned from the endpoint (sample_rag_trace)
    trace_data = mock_tracing_client._upload_trace_data.call_args[0][1]
    assert trace_data.spans[0].name == "rag"
    assert trace_data.spans[0].inputs == {"question": "query"}
    assert trace_data.spans[0].outputs == "answer"


def test_to_predict_fn_return_v2_trace(mock_deploy_client, mock_tracing_client):
    mlflow.tracing.reset()

    mock_deploy_client.predict.return_value = {
        **_DUMMY_CHAT_RESPONSE,
        "databricks_output": {"trace": V2_TRACE_DICT},
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
    mock_tracing_client.start_trace.assert_called_once()
    trace_info = mock_tracing_client.start_trace.call_args[0][0]
    # Copied trace should have a new trace ID (and v3)
    isinstance(trace_info, TraceInfo)
    assert trace_info.trace_id != V2_TRACE_DICT["info"]["request_id"]
    assert trace_info.request_preview == '{"x": 2, "y": 5}'
    assert trace_info.response_preview == "8"
    trace_data = mock_tracing_client._upload_trace_data.call_args[0][1]
    assert len(trace_data.spans) == 2
    assert trace_data.spans[0].name == "predict"
    assert trace_data.spans[0].inputs == {"x": 2, "y": 5}
    assert trace_data.spans[0].outputs == 8
    mock_tracing_client._upload_trace_data.assert_called_once_with(mock.ANY, trace_data)


def test_to_predict_fn_should_not_pass_databricks_options_to_fmapi(
    mock_deploy_client, mock_tracing_client
):
    mock_deploy_client.get_endpoint.return_value = {
        "endpoint_type": "FOUNDATION_MODEL_API",
    }
    mock_deploy_client.predict.return_value = _DUMMY_CHAT_RESPONSE
    messages = [
        {"content": "You are a helpful assistant.", "role": "system"},
        {"content": "What is Spark?", "role": "user"},
    ]

    predict_fn = to_predict_fn("endpoints:/foundation-model-api")
    response = predict_fn(messages=messages)

    mock_deploy_client.predict.assert_called_once_with(
        endpoint="foundation-model-api",
        inputs={"messages": messages},
    )
    assert response == _DUMMY_CHAT_RESPONSE  # Response should not contain databricks_output

    # Bare-minimum trace should be created when the endpoint does not return a trace
    mock_tracing_client.start_trace.assert_called_once()
    trace_info = mock_tracing_client.start_trace.call_args[0][0]
    assert trace_info.request_preview == "What is Spark?"
    trace_data = mock_tracing_client._upload_trace_data.call_args[0][1]
    assert len(trace_data.spans) == 1
    assert trace_data.spans[0].name == "predict"


def test_to_predict_fn_handles_trace_without_tags(
    sample_rag_trace, mock_deploy_client, mock_tracing_client
):
    # Create a trace dict without `tags` field
    trace_dict = sample_rag_trace.to_dict()
    trace_dict["info"].pop("tags", None)  # Remove tags field entirely

    mock_deploy_client.predict.return_value = {
        **_DUMMY_CHAT_RESPONSE,
        "databricks_output": {"trace": trace_dict},
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
    assert response == _DUMMY_CHAT_RESPONSE

    # Trace should be copied successfully even without tags
    mock_tracing_client.start_trace.assert_called_once()
    trace_info = mock_tracing_client.start_trace.call_args[0][0]
    assert trace_info.trace_id != sample_rag_trace.info.trace_id
    assert trace_info.request_preview == '{"question": "query"}'
    assert trace_info.response_preview == '"answer"'

    trace_data = mock_tracing_client._upload_trace_data.call_args[0][1]
    assert len(trace_data.spans) == 3
    mock_tracing_client._upload_trace_data.assert_called_once_with(mock.ANY, trace_data)


def test_to_predict_fn_reuses_trace_in_dual_write_mode(
    sample_rag_trace, mock_deploy_client, mock_tracing_client
):
    """
    Test that when an endpoint logs traces to both inference table and MLflow experiment
    (dual-write mode), the trace is reused instead of being re-logged.

    This happens when MLFLOW_EXPERIMENT_ID env var is set in the serving endpoint.
    """
    # Set up an experiment context
    experiment_id = "test-experiment-123"
    with mock.patch(
        "mlflow.genai.evaluation.base._get_experiment_id", return_value=experiment_id
    ) as mock_get_experiment_id:
        # Create a trace dict with experiment_id matching the current experiment
        trace_dict = sample_rag_trace.to_dict()
        trace_dict["info"]["trace_location"] = {
            "mlflow_experiment": {"experiment_id": experiment_id}
        }

        mock_deploy_client.predict.return_value = {
            **_DUMMY_CHAT_RESPONSE,
            "databricks_output": {"trace": trace_dict},
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
        assert response == _DUMMY_CHAT_RESPONSE

        # The trace should NOT be copied when it's already in the current experiment
        mock_tracing_client.start_trace.assert_not_called()
        mock_tracing_client._upload_trace_data.assert_not_called()
        mock_get_experiment_id.assert_called_once()


def test_to_predict_fn_copies_trace_when_experiment_differs(
    sample_rag_trace, mock_deploy_client, mock_tracing_client
):
    """
    Test that when an endpoint returns a trace from a different experiment,
    the trace is still copied to the current experiment.
    """
    # Set up an experiment context
    current_experiment_id = "current-experiment-123"
    endpoint_experiment_id = "different-experiment-456"

    with mock.patch(
        "mlflow.genai.evaluation.base._get_experiment_id", return_value=current_experiment_id
    ) as mock_get_experiment_id:
        # Create a trace dict with a different experiment_id
        trace_dict = sample_rag_trace.to_dict()
        trace_dict["info"]["trace_location"] = {
            "mlflow_experiment": {"experiment_id": endpoint_experiment_id}
        }

        mock_deploy_client.predict.return_value = {
            **_DUMMY_CHAT_RESPONSE,
            "databricks_output": {"trace": trace_dict},
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
        assert response == _DUMMY_CHAT_RESPONSE

        # The trace SHOULD be copied when experiments differ
        mock_tracing_client.start_trace.assert_called_once()
        trace_info = mock_tracing_client.start_trace.call_args[0][0]
        # Copied trace should have a new trace ID
        assert trace_info.trace_id != sample_rag_trace.info.trace_id
        mock_tracing_client._upload_trace_data.assert_called_once()
        mock_get_experiment_id.assert_called_once()
