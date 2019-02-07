import mock
import os
import pytest

import mlflow
from mlflow.store.dbmodels.db_types import DATABASE_ENGINES
from mlflow.store.file_store import FileStore
from mlflow.store.rest_store import RestStore
from mlflow.store.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking.utils import _get_store, _TRACKING_URI_ENV_VAR, _TRACKING_USERNAME_ENV_VAR, \
    _TRACKING_PASSWORD_ENV_VAR, _TRACKING_TOKEN_ENV_VAR, _TRACKING_INSECURE_TLS_ENV_VAR, \
    get_db_profile_from_uri, _download_artifact_from_uri, _tracking_store_registry, \
    TrackingStoreRegistry


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
    with mock.patch.dict(os.environ, env), patch_create_engine as mock_create_engine:
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

    with mock.patch.dict(os.environ, env), patch_create_engine as mock_create_engine:
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


def test_standard_store_registry():
    expected_standard_registry = {
        '',
        'file',
        'http',
        'https',
        'postgresql',
        'mysql',
        'sqlite',
        'mssql',
        'databricks'
    }
    assert expected_standard_registry.issubset(
        _tracking_store_registry._registry.keys()
    )


def test_plugin_registration():
    tracking_store = TrackingStoreRegistry()

    mock_plugin = mock.Mock()
    tracking_store.register("mock-scheme", mock_plugin)
    assert "mock-scheme" in tracking_store._registry
    assert tracking_store.get_store("mock-scheme://fake-host/fake-path") == mock_plugin.return_value


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


def test_get_db_profile_from_uri_casing():
    assert get_db_profile_from_uri('databricks://aAbB') == 'aAbB'


def test_artifact_can_be_downloaded_from_absolute_uri_successfully(tmpdir):
    artifact_file_name = "artifact.txt"
    artifact_text = "Sample artifact text"
    local_artifact_path = tmpdir.join(artifact_file_name).strpath
    with open(local_artifact_path, "w") as out:
        out.write(artifact_text)

    logged_artifact_path = "artifact"
    with mlflow.start_run():
        mlflow.log_artifact(local_path=local_artifact_path, artifact_path=logged_artifact_path)
        artifact_uri = mlflow.get_artifact_uri(artifact_path=logged_artifact_path)

    downloaded_artifact_path = os.path.join(
        _download_artifact_from_uri(artifact_uri), artifact_file_name)
    assert downloaded_artifact_path != local_artifact_path
    assert downloaded_artifact_path != logged_artifact_path
    with open(downloaded_artifact_path, "r") as f:
        assert f.read() == artifact_text


def test_download_artifact_from_absolute_uri_persists_data_to_specified_output_directory(tmpdir):
    artifact_file_name = "artifact.txt"
    artifact_text = "Sample artifact text"
    local_artifact_path = tmpdir.join(artifact_file_name).strpath
    with open(local_artifact_path, "w") as out:
        out.write(artifact_text)

    logged_artifact_subdir = "logged_artifact"
    with mlflow.start_run():
        mlflow.log_artifact(local_path=local_artifact_path, artifact_path=logged_artifact_subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path=logged_artifact_subdir)

    artifact_output_path = tmpdir.join("artifact_output").strpath
    os.makedirs(artifact_output_path)
    _download_artifact_from_uri(artifact_uri=artifact_uri, output_path=artifact_output_path)
    assert logged_artifact_subdir in os.listdir(artifact_output_path)
    assert artifact_file_name in os.listdir(
        os.path.join(artifact_output_path, logged_artifact_subdir))
    with open(os.path.join(
            artifact_output_path, logged_artifact_subdir, artifact_file_name), "r") as f:
        assert f.read() == artifact_text
