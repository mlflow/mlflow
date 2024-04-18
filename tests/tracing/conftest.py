from unittest import mock

import pytest
from opentelemetry.trace import _TRACER_PROVIDER_SET_ONCE

import mlflow
from mlflow.entities import Trace, TraceData, TraceInfo, TraceStatus
from mlflow.entities.span_status import SpanStatus
from mlflow.tracing.clients import InMemoryTraceClient, InMemoryTraceClientWithTracking
from mlflow.tracing.display import IPythonTraceDisplayHandler
from mlflow.tracing.provider import _TRACER_PROVIDER_INITIALIZED, _setup_tracer_provider
from mlflow.tracing.trace_manager import InMemoryTraceManager


@pytest.fixture(autouse=True)
def clear_singleton():
    """
    Clear the singleton instances after each tests to avoid side effects.
    """
    InMemoryTraceManager._instance = None
    InMemoryTraceClient._instance = None
    InMemoryTraceClientWithTracking._instance = None
    IPythonTraceDisplayHandler._instance = None

    # Tracer provider also needs to be reset as it may hold reference to the singleton
    with _TRACER_PROVIDER_SET_ONCE._lock:
        _TRACER_PROVIDER_SET_ONCE._done = False
    with _TRACER_PROVIDER_INITIALIZED._lock:
        _TRACER_PROVIDER_INITIALIZED._done = False


@pytest.fixture
def mock_client():
    # OpenTelemetry doesn't allow re-initializing the tracer provider within a single
    # process. However, we need to create a new tracer provider with the new mock client
    # so hack the Once object to allow re-initialization.
    from opentelemetry.trace import _TRACER_PROVIDER_SET_ONCE

    with _TRACER_PROVIDER_SET_ONCE._lock:
        _TRACER_PROVIDER_SET_ONCE._done = False

    mock_client = InMemoryTraceClientWithTracking.get_instance()

    _setup_tracer_provider(mock_client)

    yield mock_client

    # Clear traces collected in the buffer
    mock_client._flush()


@pytest.fixture
def create_trace():
    return lambda id: Trace(
        info=TraceInfo(
            request_id=id,
            experiment_id="test",
            timestamp_ms=0,
            execution_time_ms=1,
            status=TraceStatus.OK,
            request_metadata={},
            tags={},
        ),
        data=TraceData(),
    )


@pytest.fixture(autouse=True)
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None


@pytest.fixture
def mock_tracking_service_client():
    with mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient.create_trace_info",
        return_value=TraceInfo(
            request_id="tr-1234",
            experiment_id="0",
            timestamp_ms=0,
            execution_time_ms=0,
            status=SpanStatus(TraceStatus.OK),
            request_metadata={},
            tags={"mlflow.artifactLocation": "test"},
        ),
    ) as mock_create_trace_info, mock.patch(
        "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data",
        return_value=None,
    ) as mock_upload_trace_data:
        yield

        mock_create_trace_info.assert_called()
        mock_upload_trace_data.assert_called()
