import os

import pytest

import mlflow
from mlflow.utils.file_utils import path_to_local_sqlite_uri


from _pytest.terminal import TerminalReporter


class NewTerminalReporter(TerminalReporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_is_failure = False

    def summary_errors(self, *args, **kwargs):
        print("::group::ERRORS")
        super().summary_errors(*args, **kwargs)
        print("::endgroup::")

    def summary_failures(self, *args, **kwargs):
        print("::group::FAILURES")
        super().summary_errors(*args, **kwargs)
        print("::endgroup::")

    def summary_warnings(self, *args, **kwargs):
        print("::group::WARNINGS")
        super().summary_warnings(*args, **kwargs)
        print("::endgroup::")


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    import sys

    old_reporter = config.pluginmanager.getplugin("terminalreporter")
    config.pluginmanager.unregister(old_reporter)
    new_reporter = NewTerminalReporter(config, sys.stdout)
    config.pluginmanager.register(new_reporter, "terminalreporter")


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
