import os

import pytest

import mlflow
from mlflow.utils.autologging_utils import (
    _is_testing,
    _AUTOLOGGING_TEST_MODE_ENV_VAR,
)
from mlflow.utils.file_utils import path_to_local_sqlite_uri


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
    try:
        prev_env_var_value = os.environ.pop(_AUTOLOGGING_TEST_MODE_ENV_VAR, None)
        os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = "true"
        assert _is_testing()
        yield
    finally:
        if prev_env_var_value:
            os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR] = prev_env_var_value
        else:
            del os.environ[_AUTOLOGGING_TEST_MODE_ENV_VAR]
