import subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest
import requests

import mlflow
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_LOGGING,
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
)
from mlflow.tracing.fluent import _flush_pending_async_trace_writes
from mlflow.utils import find_free_port


@pytest.fixture(autouse=True)
def enable_async_trace_logging(monkeypatch):
    """Enable async trace logging for all tests in tests/tracing/ to exercise the async path.

    Overrides the global disable_async_trace_logging fixture from tests/conftest.py.
    Tests that need both sync and async coverage use the async_logging_enabled fixture.
    Terminates async queues on teardown to prevent thread leaks between tests.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.name, "true")
    monkeypatch.setenv(MLFLOW_ENABLE_ASYNC_LOGGING.name, "true")

    yield

    _flush_pending_async_trace_writes(terminate=True)


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


# Fixture to run the test case with and without async logging enabled.
# When async logging is enabled, the batch span processor is also active (the default),
# so tests exercise the full production pipeline.
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
    for attempt in range(3):
        try:
            subprocess.check_call(["docker", "pull", "otel/opentelemetry-collector"])
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))

    port = find_free_port()
    while (health_port := find_free_port()) == port:
        pass

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

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [health_check]
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [debug]"""

    with tempfile.NamedTemporaryFile() as output_file:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-p",
            f"127.0.0.1:{port}:4317",
            "-p",
            f"127.0.0.1:{health_port}:13133",
            "-i",
            "otel/opentelemetry-collector",
            "--config=/dev/stdin",
        ]

        process = subprocess.Popen(
            docker_cmd,
            stdin=subprocess.PIPE,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            try:
                process.stdin.write(docker_collector_config)
                process.stdin.close()
            except BrokenPipeError:
                # The readiness loop below reports collector exit with logs.
                pass

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    logs = Path(output_file.name).read_text()
                    raise RuntimeError(f"Collector exited ({process.returncode}):\n{logs}")
                try:
                    response = requests.get(f"http://127.0.0.1:{health_port}/", timeout=1)
                except requests.RequestException:
                    pass
                else:
                    if response.status_code == 200:
                        break
                time.sleep(0.1)
            else:
                logs = Path(output_file.name).read_text()
                raise TimeoutError(f"Collector did not become healthy within 10 seconds:\n{logs}")

            yield output_file.name, port
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
