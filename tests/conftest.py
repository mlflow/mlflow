import os

import pytest

import mlflow

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
    from mlflow.utils.file_utils import path_to_local_sqlite_uri

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
    test_mode_on()


@pytest.fixture(scope="function")
def sqlite_store():
    fd, temp_dbfile = tempfile.mkstemp()
    # Close handle immediately so that we can remove the file later on in Windows
    os.close(fd)
    db_uri = "sqlite:///%s" % temp_dbfile
    store = SqlAlchemyStore(db_uri, "artifact_folder")
    yield (store, db_uri)
    os.remove(temp_dbfile)
    shutil.rmtree("artifact_folder")


@pytest.fixture(scope="function")
def file_store():
    ROOT_LOCATION = os.path.join(tempfile.gettempdir(), "test_mlflow_gc")
    file_store_uri = "file:///%s" % ROOT_LOCATION
    yield (FileStore(ROOT_LOCATION), file_store_uri)
    shutil.rmtree(ROOT_LOCATION)
