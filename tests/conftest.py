import os

import pytest

import mlflow
from mlflow.utils.file_utils import path_to_local_sqlite_uri

from tests.autologging.fixtures import test_mode_on


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
    yield from test_mode_on()


ALL_TESTS_BEFORE = []
ALL_TESTS_AFTER = []


@pytest.fixture(autouse=True)
def clean_up_leaked_runs():
    """
    Certain test cases validate safety API behavior when runs are leaked. Leaked runs that
    are not cleaned up between test cases may result in cascading failures that are hard to
    debug. Accordingly, this fixture attempts to end any active runs it encounters and
    throws an exception (which reported as an additional error in the pytest execution output).
    """
    try:
        import os
        test_info = os.environ.get('PYTEST_CURRENT_TEST')
        ALL_TESTS_BEFORE.append(test_info)
        yield
        ALL_TESTS_AFTER.append(test_info)
        assert not mlflow.active_run(), "test case unexpectedly leaked a run!: " + mlflow.get_tracking_uri() + " BEFORE: " + ",".join(ALL_TESTS_BEFORE) + " AFTER : " + ",".join(ALL_TESTS_AFTER)
    finally:
        while mlflow.active_run():
            mlflow.end_run()
