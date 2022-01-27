import os
import inspect
import shutil
import subprocess
from unittest import mock

import pytest

import mlflow
from mlflow.utils.file_utils import path_to_local_sqlite_uri

from tests.autologging.fixtures import enable_test_mode


@pytest.fixture
def reset_mock():
    cache = []

    def set_mock(obj, attr, mock):
        cache.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, mock)

    yield set_mock

    for obj, attr, value in cache:
        setattr(obj, attr, value)
    cache[:] = []


@pytest.fixture(autouse=True)
def tracking_uri_mock(tmpdir, request):
    try:
        if "notrackingurimock" not in request.keywords:
            tracking_uri = path_to_local_sqlite_uri(os.path.join(tmpdir.strpath, "mlruns"))
            mlflow.set_tracking_uri(tracking_uri)
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        yield tmpdir
    finally:
        mlflow.set_tracking_uri(None)
        if "notrackingurimock" not in request.keywords:
            del os.environ["MLFLOW_TRACKING_URI"]


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
        assert (
            not mlflow.active_run()
        ), "test case unexpectedly leaked a run. Run info: {}. Run data: {}".format(
            mlflow.active_run().info, mlflow.active_run().data
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
    from mlflow.utils.environment import _INFER_PIP_REQUIREMENTS_FALLBACK_MESSAGE

    def new_exception(msg, *_, **__):
        if msg == _INFER_PIP_REQUIREMENTS_FALLBACK_MESSAGE and _called_in_save_model():
            raise Exception(
                "`mlflow.models.infer_pip_requirements` should not fall back in"
                "`mlflow.*.save_model` during test"
            )

    if "allow_infer_pip_requirements_fallback" not in request.keywords:
        with mock.patch("mlflow.utils.environment._logger.exception", new=new_exception):
            yield
    else:
        yield


@pytest.fixture(autouse=True, scope="module")
def clean_up_mlruns_direcotry(request):
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
        except IOError:
            if os.name == "nt":
                raise
            # `shutil.rmtree` can't remove files owned by root in a docker container.
            subprocess.run(["sudo", "rm", "-rf", mlruns_dir], check=True)
