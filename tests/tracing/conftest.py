import random
import subprocess
import tempfile
import time
from unittest import mock

import pytest

import mlflow
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_LOGGING,
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
)


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
    subprocess.run(["docker", "pull", "otel/opentelemetry-collector"], check=True)

    # Use a random port to avoid conflicts
    port = random.randint(20000, 30000)

    docker_collector_config = """receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  debug:
    verbosity: detailed
    sampling_initial: 5
    sampling_thereafter: 1

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [debug]"""

    with tempfile.NamedTemporaryFile() as output_file:
        # Use echo to pipe config to Docker stdin
        docker_cmd = [
            "bash",
            "-c",
            f'echo "{docker_collector_config}" | '
            f"docker run --rm -p 127.0.0.1:{port}:4317 -i "
            f"otel/opentelemetry-collector --config=/dev/stdin",
        ]

        process = subprocess.Popen(
            docker_cmd,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for the collector to start
        time.sleep(5)

        yield process, output_file.name, port

        # Stop the collector
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
