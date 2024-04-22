from unittest import mock

from opentelemetry import trace

from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import RestException
from mlflow.tracing.clients import get_trace_client
from mlflow.tracing.provider import _TRACER_PROVIDER_INITIALIZED, _get_tracer, create_trace_info
from mlflow.tracing.utils import encode_trace_id

from tests.tracing.helper import create_mock_otel_span


# Mock client getter just to count the number of calls
@mock.patch("mlflow.tracing.provider.get_trace_client", side_effect=get_trace_client)
def test_tracer_provider_singleton(mock_get_client):
    # Reset the Once object as there might be other tests that have already initialized it
    _TRACER_PROVIDER_INITIALIZED._done = False

    _get_tracer("module_1")
    assert mock_get_client.call_count == 1
    assert _TRACER_PROVIDER_INITIALIZED._done is True

    tracer_provider_1 = trace.get_tracer_provider()

    _get_tracer("module_2")
    assert mock_get_client.call_count == 1

    tracer_provider_2 = trace.get_tracer_provider()

    # Trace provider should be identical for different moments in time
    assert tracer_provider_1 is tracer_provider_2


def test_create_trace_info_databricks(monkeypatch, mock_store, databricks_tracking_uri):
    otel_span = create_mock_otel_span(trace_id=111, span_id=1, start_time=123456789)
    trace_info = create_trace_info(
        otel_span=otel_span,
        experiment_id="test_experiment_id",
        request_metadata={"key": "value"},
        # Trying to override mlflow.user tag
        tags={"foo": "bar", "mlflow.user": "alice"},
    )

    mock_store.start_trace.assert_called_once()
    assert trace_info.request_id == "tr-12345"
    assert trace_info.experiment_id == "test_experiment_id"
    assert trace_info.timestamp_ms == 123
    assert trace_info.execution_time_ms is None
    assert trace_info.status == TraceStatus.IN_PROGRESS
    assert trace_info.request_metadata == {"key": "value"}
    # mlflow.user tag should be overridden by the one passed by the user
    assert trace_info.tags == {
        "foo": "bar",
        "mlflow.user": "bob",
        "mlflow.artifactLocation": "test",
    }


def test_create_trace_info_databricks_get_experiemnt_id_from_env(
    monkeypatch, mock_store, databricks_tracking_uri
):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "1")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test")
    mock_store.get_experiment_by_name().experiment_id = "test_experiment_id"

    otel_span = create_mock_otel_span(trace_id=111, span_id=1, start_time=123456789)
    trace_info = create_trace_info(otel_span=otel_span)

    mock_store.start_trace.assert_called_once()
    assert trace_info.request_id == "tr-12345"
    assert trace_info.experiment_id == "test_experiment_id"
    assert trace_info.timestamp_ms == 123


def test_create_trace_info_fallback_to_local_when_rest_exception_raised(
    monkeypatch, mock_store, databricks_tracking_uri
):
    mock_store.start_trace.side_effect = RestException({"error_code": "RESOURCE_DOES_NOT_EXIST"})

    otel_span = create_mock_otel_span(trace_id=111, span_id=1, start_time=123456789)
    trace_info = create_trace_info(otel_span=otel_span, experiment_id="test_experiment_id")

    mock_store.start_trace.assert_called_once()
    assert trace_info.request_id == encode_trace_id(111)
    assert trace_info.experiment_id == "test_experiment_id"
    assert trace_info.timestamp_ms == 123
    assert trace_info.execution_time_ms is None
    assert trace_info.status == TraceStatus.IN_PROGRESS
    assert trace_info.request_metadata == {}
    assert trace_info.tags == {}
