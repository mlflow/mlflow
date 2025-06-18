import subprocess
import tempfile
import time
from unittest import mock

from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
import pytest

import mlflow
from mlflow.entities import TraceInfoV2
from mlflow.entities.trace_status import TraceStatus
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING, MLFLOW_ENABLE_ASYNC_TRACE_LOGGING


@pytest.fixture(autouse=True)
def reset_active_experiment():
    yield
    mlflow.tracking.fluent._active_experiment_id = None


@pytest.fixture(autouse=True)
def reset_tracking_uri():
    # Some API like set_destination("databricks") updates the tracking URI,
    # we should reset it between tests
    original_tracking_uri = mlflow.get_tracking_uri()

    yield

    mlflow.set_tracking_uri(original_tracking_uri)


@pytest.fixture
def mock_upload_trace_data():
    with (
        mock.patch(
            "mlflow.tracking._tracking_service.client.TrackingServiceClient.end_trace",
            return_value=TraceInfoV2(
                request_id="tr-1234",
                experiment_id="0",
                timestamp_ms=0,
                execution_time_ms=0,
                status=TraceStatus.OK,
                request_metadata={},
                tags={"mlflow.artifactLocation": "test"},
            ),
        ),
        mock.patch(
            "mlflow.tracking._tracking_service.client.TrackingServiceClient._upload_trace_data",
            return_value=None,
        ) as mock_upload_trace_data,
    ):
        yield mock_upload_trace_data


@pytest.fixture
def databricks_tracking_uri():
    with mock.patch("mlflow.get_tracking_uri", return_value="databricks"):
        yield


# Fixture to run the test case with and without async logging enabled
@pytest.fixture(params=[True, False])
def async_logging_enabled(request, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, str(request.param))
    # TODO: V2 Trace depends on this env var rather than MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
    # We should remove this once the backend is fully migrated to V3
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_LOGGING.name, str(request.param))
    return request.param


@pytest.fixture
def otel_collector():
    """Start an OpenTelemetry collector in a Docker container."""
    subprocess.run(["docker", "pull", "otel/opentelemetry-collector-contrib"], check=True)

    with (
        tempfile.NamedTemporaryFile() as output_file,
        subprocess.Popen(
            [
                "docker",
                "run",
                "-p",
                "127.0.0.1:4317:4317",
                "otel/opentelemetry-collector",
            ],
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process,
    ):
        # Wait for the collector to start
        time.sleep(5)

        yield process, output_file.name

        # Stop the collector
        container_id = subprocess.check_output(
            ["docker", "ps", "-q", "--filter", "ancestor=otel/opentelemetry-collector"],
            text=True,
        ).strip()
        subprocess.check_call(["docker", "stop", container_id])
