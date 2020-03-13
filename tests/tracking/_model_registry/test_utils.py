import mock
import os

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.tracking._model_registry.utils import _get_store
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR

# Disable mocking tracking URI here, as we want to test setting the tracking URI via
# environment variable. See
# http://doc.pytest.org/en/latest/skipping.html#skip-all-test-functions-of-a-class-or-module
# and https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.rst#writing-python-tests
# for more information.
pytestmark = pytest.mark.notrackingurimock


def test_get_store_rest_store_from_arg():
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050"  # should be ignored
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store("http://some/path")
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "http://some/path"


def test_fallback_to_tracking_store():
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050"
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "https://my-tracking-server:5050"
        assert store.get_host_creds().token is None


@pytest.mark.parametrize("db_type", DATABASE_ENGINES)
def test_get_store_sqlalchemy_store(db_type):
    patch_create_engine = mock.patch("sqlalchemy.create_engine")
    uri = "{}://hostname/database".format(db_type)
    env = {
        _TRACKING_URI_ENV_VAR: uri
    }

    with mock.patch.dict(os.environ, env), patch_create_engine as mock_create_engine, \
            mock.patch("mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore."
                       "_verify_registry_tables_exist"):
        store = _get_store()
        assert isinstance(store, SqlAlchemyStore)
        assert store.db_uri == uri

    mock_create_engine.assert_called_once_with(uri, pool_pre_ping=True)


@pytest.mark.parametrize("bad_uri", ["badsql://imfake", "yoursql://hi"])
def test_get_store_bad_uris(bad_uri):
    env = {
        _TRACKING_URI_ENV_VAR: bad_uri
    }

    with mock.patch.dict(os.environ, env), pytest.raises(MlflowException):
        _get_store()
