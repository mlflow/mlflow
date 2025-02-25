import inspect
import os
import shutil
import subprocess
import uuid
from unittest import mock

import pytest
from opentelemetry import trace as trace_api

import mlflow
from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler
from mlflow.tracing.export.inference_table import _TRACE_BUFFER
from mlflow.tracing.fluent import TRACE_BUFFER
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.file_utils import path_to_local_sqlite_uri
from mlflow.utils.os import is_windows

from tests.autologging.fixtures import enable_test_mode


@pytest.fixture(autouse=True)
def tracking_uri_mock(tmp_path, request):
    if "notrackingurimock" not in request.keywords:
        tracking_uri = path_to_local_sqlite_uri(tmp_path / f"{uuid.uuid4().hex}.sqlite")
        with _use_tracking_uri(tracking_uri):
            yield tracking_uri
    else:
        yield None


@pytest.fixture(autouse=True)
def reset_active_experiment_id():
    yield
    mlflow.tracking.fluent._active_experiment_id = None
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)


@pytest.fixture(autouse=True)
def reset_mlflow_uri():
    yield
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_REGISTRY_URI", None)


@pytest.fixture(autouse=True)
def reset_tracing():
    """
    Reset the global state of the tracing feature.

    This fixture is auto-applied for cleaning up the global state between tests
    to avoid side effects.
    """
    yield

    # Reset OpenTelemetry and MLflow tracer setup
    mlflow.tracing.reset()

    # Clear other global state and singletons
    TRACE_BUFFER.clear()
    _TRACE_BUFFER.clear()
    InMemoryTraceManager.reset()
    IPythonTraceDisplayHandler._instance = None


def _is_span_active():
    span = trace_api.get_current_span()
    return (span is not None) and not isinstance(span, trace_api.NonRecordingSpan)


@pytest.fixture(autouse=True)
def validate_trace_finish():
    """
    Validate all spans are finished and detached from the context by the end of the each test.

    Leaked span is critical problem and also hard to find without an explicit check.
    """
    # When the span is leaked, it causes confusing test failure in the subsequent tests. To avoid
    # this and make the test failure more clear, we fail first here.
    if _is_span_active():
        pytest.skip(reason="A leaked active span is found before starting the test.")

    yield

    assert not _is_span_active(), (
        "A span is still active at the end of the test. All spans must be finished "
        "and detached from the context before the test ends. The leaked span context "
        "may cause other subsequent tests to fail."
    )


@pytest.fixture(autouse=True, scope="session")
def enable_test_mode_by_default_for_autologging_integrations():
    """
    Run all MLflow tests in autologging test mode, ensuring that errors in autologging patch code
    are raised and detected. For more information about autologging test mode, see the docstring
    for :py:func:`mlflow.utils.autologging_utils._is_testing()`.
    """
    yield from enable_test_mode()


@pytest.fixture(autouse=True)
def clean_up_leaked_runs():
    """
    Certain test cases validate safety API behavior when runs are leaked. Leaked runs that
    are not cleaned up between test cases may result in cascading failures that are hard to
    debug. Accordingly, this fixture attempts to end any active runs it encounters and
    throws an exception (which reported as an additional error in the pytest execution output).
    """
    try:
        yield
        assert not mlflow.active_run(), (
            "test case unexpectedly leaked a run. Run info: {}. Run data: {}".format(
                mlflow.active_run().info, mlflow.active_run().data
            )
        )
    finally:
        while mlflow.active_run():
            mlflow.end_run()


def _called_in_save_model():
    for frame in inspect.stack()[::-1]:
        if frame.function == "save_model":
            return True
    return False


@pytest.fixture(autouse=True)
def prevent_infer_pip_requirements_fallback(request):
    """
    Prevents `mlflow.models.infer_pip_requirements` from falling back in `mlflow.*.save_model`
    unless explicitly disabled via `pytest.mark.allow_infer_pip_requirements_fallback`.
    """
    from mlflow.utils.environment import _INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE

    def new_exception(msg, *_, **__):
        if msg == _INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE and _called_in_save_model():
            raise Exception(
                "`mlflow.models.infer_pip_requirements` should not fall back in"
                "`mlflow.*.save_model` during test"
            )

    if "allow_infer_pip_requirements_fallback" not in request.keywords:
        with mock.patch("mlflow.utils.environment._logger.exception", new=new_exception):
            yield
    else:
        yield


@pytest.fixture(autouse=True)
def clean_up_mlruns_directory(request):
    """
    Clean up an `mlruns` directory on each test module teardown on CI to save the disk space.
    """
    yield

    # Only run this fixture on CI.
    if "GITHUB_ACTIONS" not in os.environ:
        return

    mlruns_dir = os.path.join(request.config.rootpath, "mlruns")
    if os.path.exists(mlruns_dir):
        try:
            shutil.rmtree(mlruns_dir)
        except OSError:
            if is_windows():
                raise
            # `shutil.rmtree` can't remove files owned by root in a docker container.
            subprocess.run(["sudo", "rm", "-rf", mlruns_dir], check=True)


@pytest.fixture
def mock_s3_bucket():
    """
    Creates a mock S3 bucket using moto

    Returns:
        The name of the mock bucket.
    """
    import boto3
    import moto

    with moto.mock_s3():
        bucket_name = "mock-bucket"
        s3_client = boto3.client("s3")
        s3_client.create_bucket(Bucket=bucket_name)
        yield bucket_name


class ExtendedMonkeyPatch(pytest.MonkeyPatch):  # type: ignore
    def setenvs(self, envs, prepend=None):
        for name, value in envs.items():
            self.setenv(name, value, prepend)

    def delenvs(self, names, raising=True):
        for name in names:
            self.delenv(name, raising)


@pytest.fixture
def monkeypatch():
    """
    Overrides the default monkeypatch fixture to use `ExtendedMonkeyPatch`.
    """
    with ExtendedMonkeyPatch().context() as mp:
        yield mp


@pytest.fixture
def tmp_sqlite_uri(tmp_path):
    path = tmp_path.joinpath("mlflow.db").as_uri()
    return ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]


@pytest.fixture
def mock_databricks_serving_with_tracing_env(monkeypatch):
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv("ENABLE_MLFLOW_TRACING", "true")


@pytest.fixture(params=[True, False])
def mock_is_in_databricks(request):
    with mock.patch(
        "mlflow.models.model.is_in_databricks_runtime", return_value=request.param
    ) as mock_databricks:
        yield mock_databricks
