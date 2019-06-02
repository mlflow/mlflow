import mock
import os
import pytest
from six.moves import reload_module as reload

import mlflow
from mlflow.store.dbmodels.db_types import DATABASE_ENGINES
from mlflow.store.file_store import FileStore
from mlflow.store.rest_store import RestStore
from mlflow.store.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking.utils import _get_store, _TRACKING_URI_ENV_VAR, _TRACKING_USERNAME_ENV_VAR, \
    _TRACKING_PASSWORD_ENV_VAR, _TRACKING_TOKEN_ENV_VAR, _TRACKING_INSECURE_TLS_ENV_VAR, \
    get_db_profile_from_uri, TrackingStoreRegistry


def test_get_store_file_store(tmp_wkdir):
    env = {}
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, FileStore)
        assert os.path.abspath(store.root_directory) == os.path.abspath("mlruns")


def test_get_store_file_store_from_arg(tmp_wkdir):
    env = {}
    with mock.patch.dict(os.environ, env):
        store = _get_store("other/path")
        assert isinstance(store, FileStore)
        assert os.path.abspath(store.root_directory) == os.path.abspath("other/path")


@pytest.mark.parametrize("uri", ["other/path", "file:other/path"])
def test_get_store_file_store_from_env(tmp_wkdir, uri):
    env = {
        _TRACKING_URI_ENV_VAR: uri
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, FileStore)
        assert os.path.abspath(store.root_directory) == os.path.abspath("other/path")


def test_get_store_basic_rest_store():
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050"
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "https://my-tracking-server:5050"
        assert store.get_host_creds().token is None


def test_get_store_rest_store_with_password():
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_USERNAME_ENV_VAR: "Bob",
        _TRACKING_PASSWORD_ENV_VAR: "Ross",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "https://my-tracking-server:5050"
        assert store.get_host_creds().username == "Bob"
        assert store.get_host_creds().password == "Ross"


def test_get_store_rest_store_with_token():
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_TOKEN_ENV_VAR: "my-token",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().token == "my-token"


def test_get_store_rest_store_with_insecure():
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_INSECURE_TLS_ENV_VAR: "true",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().ignore_tls_verification


def test_get_store_rest_store_with_no_insecure():
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_INSECURE_TLS_ENV_VAR: "false",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert not store.get_host_creds().ignore_tls_verification

    # By default, should not ignore verification.
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert not store.get_host_creds().ignore_tls_verification


@pytest.mark.parametrize("db_type", DATABASE_ENGINES)
def test_get_store_sqlalchemy_store(tmp_wkdir, db_type):
    patch_create_engine = mock.patch("sqlalchemy.create_engine")

    uri = "{}://hostname/database".format(db_type)
    env = {
        _TRACKING_URI_ENV_VAR: uri
    }
    with mock.patch.dict(os.environ, env), patch_create_engine as mock_create_engine,\
            mock.patch("mlflow.store.sqlalchemy_store.SqlAlchemyStore._verify_schema"), \
            mock.patch("mlflow.store.sqlalchemy_store.SqlAlchemyStore._initialize_tables"):
        store = _get_store()
        assert isinstance(store, SqlAlchemyStore)
        assert store.db_uri == uri
        assert store.artifact_root_uri == "./mlruns"

    mock_create_engine.assert_called_once_with(uri)


@pytest.mark.parametrize("db_type", DATABASE_ENGINES)
def test_get_store_sqlalchemy_store_with_artifact_uri(tmp_wkdir, db_type):
    patch_create_engine = mock.patch("sqlalchemy.create_engine")
    uri = "{}://hostname/database".format(db_type)
    env = {
        _TRACKING_URI_ENV_VAR: uri
    }
    artifact_uri = "file:artifact/path"

    with mock.patch.dict(os.environ, env), patch_create_engine as mock_create_engine, \
            mock.patch("mlflow.store.sqlalchemy_store.SqlAlchemyStore._verify_schema"), \
            mock.patch("mlflow.store.sqlalchemy_store.SqlAlchemyStore._initialize_tables"):
        store = _get_store(artifact_uri=artifact_uri)
        assert isinstance(store, SqlAlchemyStore)
        assert store.db_uri == uri
        assert store.artifact_root_uri == artifact_uri

    mock_create_engine.assert_called_once_with(uri)


def test_get_store_databricks():
    env = {
        _TRACKING_URI_ENV_VAR: "databricks",
        'DATABRICKS_HOST': "https://my-tracking-server",
        'DATABRICKS_TOKEN': "abcdef",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "https://my-tracking-server"
        assert store.get_host_creds().token == "abcdef"


def test_get_store_databricks_profile():
    env = {
        _TRACKING_URI_ENV_VAR: "databricks://mycoolprofile",
    }
    # It's kind of annoying to setup a profile, and we're not really trying to test
    # that anyway, so just check if we raise a relevant exception.
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        with pytest.raises(Exception) as e_info:
            store.get_host_creds()
        assert 'mycoolprofile' in str(e_info.value)


def test_standard_store_registry_with_mocked_entrypoint():
    mock_entrypoint = mock.Mock()
    mock_entrypoint.name = "mock-scheme"

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ):
        # Entrypoints are registered at import time, so we need to reload the
        # module to register the entrypoint given by the mocked
        # extrypoints.get_group_all
        reload(mlflow.tracking.utils)

        expected_standard_registry = {
            '',
            'file',
            'http',
            'https',
            'postgresql',
            'mysql',
            'sqlite',
            'mssql',
            'databricks',
            'mock-scheme'
        }
        assert expected_standard_registry.issubset(
            mlflow.tracking.utils._tracking_store_registry._registry.keys()
        )


@pytest.mark.large
def test_standard_store_registry_with_installed_plugin(tmp_wkdir):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(mlflow.tracking.utils)
    assert "file-plugin" in mlflow.tracking.utils._tracking_store_registry._registry.keys()

    from mlflow_test_plugin import PluginFileStore

    env = {
        _TRACKING_URI_ENV_VAR: "file-plugin:test-path",
    }
    with mock.patch.dict(os.environ, env):
        plugin_file_store = mlflow.tracking.utils._get_store()
        assert isinstance(plugin_file_store, PluginFileStore)
        assert plugin_file_store.is_plugin


def test_plugin_registration():
    tracking_store = TrackingStoreRegistry()

    test_uri = "mock-scheme://fake-host/fake-path"
    test_scheme = "mock-scheme"

    mock_plugin = mock.Mock()
    tracking_store.register(test_scheme, mock_plugin)
    assert test_scheme in tracking_store._registry
    assert tracking_store.get_store(test_uri) == mock_plugin.return_value
    mock_plugin.assert_called_once_with(store_uri=test_uri, artifact_uri=None)


def test_plugin_registration_via_entrypoints():
    mock_plugin_function = mock.Mock()
    mock_entrypoint = mock.Mock(load=mock.Mock(return_value=mock_plugin_function))
    mock_entrypoint.name = "mock-scheme"

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:

        tracking_store = TrackingStoreRegistry()
        tracking_store.register_entrypoints()

    assert tracking_store.get_store("mock-scheme://") == mock_plugin_function.return_value

    mock_plugin_function.assert_called_once_with(store_uri="mock-scheme://", artifact_uri=None)
    mock_get_group_all.assert_called_once_with("mlflow.tracking_store")


@pytest.mark.parametrize("exception",
                         [AttributeError("test exception"),
                          ImportError("test exception")])
def test_handle_plugin_registration_failure_via_entrypoints(exception):
    mock_entrypoint = mock.Mock(load=mock.Mock(side_effect=exception))
    mock_entrypoint.name = "mock-scheme"

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:

        tracking_store = TrackingStoreRegistry()

        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            tracking_store.register_entrypoints()

    mock_entrypoint.load.assert_called_once()
    mock_get_group_all.assert_called_once_with("mlflow.tracking_store")


def test_get_store_for_unregistered_scheme():

    tracking_store = TrackingStoreRegistry()

    with pytest.raises(mlflow.exceptions.MlflowException,
                       match="Could not find a registered tracking store"):
        tracking_store.get_store("unknown-scheme://")


def test_get_db_profile_from_uri_casing():
    assert get_db_profile_from_uri('databricks://aAbB') == 'aAbB'
