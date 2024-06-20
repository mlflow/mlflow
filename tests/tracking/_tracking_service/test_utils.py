import io
import itertools
import os
import pickle
from importlib import reload
from pathlib import Path
from unittest import mock

import pytest

import mlflow
from mlflow.environment_variables import (
    MLFLOW_TRACKING_INSECURE_TLS,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_TOKEN,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.tracking._tracking_service.utils import (
    _get_store,
    _resolve_tracking_uri,
    get_tracking_uri,
    set_tracking_uri,
)
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows

from tests.tracing.helper import get_tracer_tracking_uri

# Disable mocking tracking URI here, as we want to test setting the tracking URI via
# environment variable. See
# http://doc.pytest.org/en/latest/skipping.html#skip-all-test-functions-of-a-class-or-module
# and https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#writing-python-tests
# for more information.
pytestmark = pytest.mark.notrackingurimock


def test_get_store_file_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = _get_store()
    assert isinstance(store, FileStore)
    assert os.path.abspath(store.root_directory) == os.path.abspath("mlruns")


def test_get_store_file_store_from_arg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = _get_store("other/path")
    assert isinstance(store, FileStore)
    assert os.path.abspath(store.root_directory) == os.path.abspath("other/path")


@pytest.mark.parametrize("uri", ["other/path", "file:other/path"])
def test_get_store_file_store_from_env(tmp_path, monkeypatch, uri):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, uri)
    store = _get_store()
    assert isinstance(store, FileStore)
    assert os.path.abspath(store.root_directory) == os.path.abspath("other/path")


def test_get_store_basic_rest_store(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "https://my-tracking-server:5050")
    store = _get_store()
    assert isinstance(store, RestStore)
    assert store.get_host_creds().host == "https://my-tracking-server:5050"
    assert store.get_host_creds().token is None


def test_get_store_rest_store_with_password(monkeypatch):
    for k, v in {
        MLFLOW_TRACKING_URI.name: "https://my-tracking-server:5050",
        MLFLOW_TRACKING_USERNAME.name: "Bob",
        MLFLOW_TRACKING_PASSWORD.name: "Ross",
    }.items():
        monkeypatch.setenv(k, v)

    store = _get_store()
    assert isinstance(store, RestStore)
    assert store.get_host_creds().host == "https://my-tracking-server:5050"
    assert store.get_host_creds().username == "Bob"
    assert store.get_host_creds().password == "Ross"


def test_get_store_rest_store_with_token(monkeypatch):
    for k, v in {
        MLFLOW_TRACKING_URI.name: "https://my-tracking-server:5050",
        MLFLOW_TRACKING_TOKEN.name: "my-token",
    }.items():
        monkeypatch.setenv(k, v)

    store = _get_store()
    assert isinstance(store, RestStore)
    assert store.get_host_creds().token == "my-token"


def test_get_store_rest_store_with_insecure(monkeypatch):
    for k, v in {
        MLFLOW_TRACKING_URI.name: "https://my-tracking-server:5050",
        MLFLOW_TRACKING_INSECURE_TLS.name: "true",
    }.items():
        monkeypatch.setenv(k, v)
    store = _get_store()
    assert isinstance(store, RestStore)
    assert store.get_host_creds().ignore_tls_verification


def test_get_store_rest_store_with_no_insecure(monkeypatch):
    with monkeypatch.context() as m:
        for k, v in {
            MLFLOW_TRACKING_URI.name: "https://my-tracking-server:5050",
            MLFLOW_TRACKING_INSECURE_TLS.name: "false",
        }.items():
            m.setenv(k, v)
        store = _get_store()
        assert isinstance(store, RestStore)
        assert not store.get_host_creds().ignore_tls_verification

    # By default, should not ignore verification.
    with monkeypatch.context() as m:
        monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "https://my-tracking-server:5050")
        store = _get_store()
        assert isinstance(store, RestStore)
        assert not store.get_host_creds().ignore_tls_verification


@pytest.mark.parametrize("db_type", DATABASE_ENGINES)
def test_get_store_sqlalchemy_store(tmp_path, monkeypatch, db_type):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", raising=False)
    patch_create_engine = mock.patch("sqlalchemy.create_engine")

    uri = f"{db_type}://hostname/database"
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, uri)
    with patch_create_engine as mock_create_engine, mock.patch(
        "mlflow.store.db.utils._verify_schema"
    ), mock.patch("mlflow.store.db.utils._initialize_tables"), mock.patch(
        # In sqlalchemy 1.4.0, `SqlAlchemyStore.search_experiments`, which is called when fetching
        # the store, results in an error when called with a mocked sqlalchemy engine.
        # Accordingly, we mock `SqlAlchemyStore.search_experiments`
        "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore.search_experiments",
        return_value=[],
    ):
        store = _get_store()
        assert isinstance(store, SqlAlchemyStore)
        assert store.db_uri == uri
        if is_windows():
            assert store.artifact_root_uri == Path.cwd().joinpath("mlruns").as_uri()
        else:
            assert store.artifact_root_uri == Path.cwd().joinpath("mlruns").as_posix()

    mock_create_engine.assert_called_once_with(uri, pool_pre_ping=True)


@pytest.mark.parametrize("db_type", DATABASE_ENGINES)
def test_get_store_sqlalchemy_store_with_artifact_uri(tmp_path, monkeypatch, db_type):
    monkeypatch.chdir(tmp_path)
    uri = f"{db_type}://hostname/database"
    artifact_uri = "file:artifact/path"
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, uri)
    with mock.patch(
        "sqlalchemy.create_engine",
    ) as mock_create_engine, mock.patch("mlflow.store.db.utils._verify_schema"), mock.patch(
        "mlflow.store.db.utils._initialize_tables"
    ), mock.patch(
        "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore.search_experiments",
        return_value=[],
    ):
        store = _get_store(artifact_uri=artifact_uri)
        assert isinstance(store, SqlAlchemyStore)
        assert store.db_uri == uri
        if is_windows():
            assert store.artifact_root_uri == Path.cwd().joinpath("artifact", "path").as_uri()
        else:
            assert store.artifact_root_uri == path_to_local_file_uri(
                Path.cwd().joinpath("artifact", "path")
            )

    mock_create_engine.assert_not_called()


def test_get_store_databricks(monkeypatch):
    for k, v in {
        MLFLOW_TRACKING_URI.name: "databricks",
        "DATABRICKS_HOST": "https://my-tracking-server",
        "DATABRICKS_TOKEN": "abcdef",
    }.items():
        monkeypatch.setenv(k, v)
    store = _get_store()
    assert isinstance(store, RestStore)
    assert store.get_host_creds().use_databricks_sdk


def test_get_store_databricks_profile(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "databricks://mycoolprofile")
    # It's kind of annoying to setup a profile, and we're not really trying to test
    # that anyway, so just check if we raise a relevant exception.
    store = _get_store()
    assert isinstance(store, RestStore)
    with pytest.raises(MlflowException, match="mycoolprofile"):
        store.get_host_creds()


def test_get_store_caches_on_store_uri_and_artifact_uri(tmp_path):
    registry = mlflow.tracking._tracking_service.utils._tracking_store_registry

    store_uri_1 = f"sqlite:///{tmp_path.joinpath('backend_store_1.db')}"
    store_uri_2 = f"sqlite:///{tmp_path.joinpath('backend_store_2.db')}"
    stores_uris = [store_uri_1, store_uri_2]
    artifact_uris = [
        None,
        str(tmp_path.joinpath("artifact_root_1")),
        str(tmp_path.joinpath("artifact_root_2")),
    ]

    stores = []
    for args in itertools.product(stores_uris, artifact_uris):
        store1 = registry.get_store(*args)
        store2 = registry.get_store(*args)
        assert store1 is store2
        stores.append(store1)

    assert all(s1 is not s2 for s1, s2 in itertools.combinations(stores, 2))


def test_standard_store_registry_with_mocked_entrypoint():
    mock_entrypoint = mock.Mock()
    mock_entrypoint.name = "mock-scheme"

    with mock.patch("entrypoints.get_group_all", return_value=[mock_entrypoint]):
        # Entrypoints are registered at import time, so we need to reload the
        # module to register the entrypoint given by the mocked
        # entrypoints.get_group_all
        reload(mlflow.tracking._tracking_service.utils)

        expected_standard_registry = {
            "",
            "file",
            "http",
            "https",
            "postgresql",
            "mysql",
            "sqlite",
            "mssql",
            "databricks",
            "mock-scheme",
        }
        assert expected_standard_registry.issubset(
            mlflow.tracking._tracking_service.utils._tracking_store_registry._registry.keys()
        )


def test_standard_store_registry_with_installed_plugin(tmp_path, monkeypatch):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""
    monkeypatch.chdir(tmp_path)
    reload(mlflow.tracking._tracking_service.utils)
    assert (
        "file-plugin" in mlflow.tracking._tracking_service.utils._tracking_store_registry._registry
    )

    from mlflow_test_plugin.file_store import PluginFileStore

    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "file-plugin:test-path")
    plugin_file_store = mlflow.tracking._tracking_service.utils._get_store()
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


@pytest.mark.parametrize(
    "exception", [AttributeError("test exception"), ImportError("test exception")]
)
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

    with pytest.raises(
        UnsupportedModelRegistryStoreURIException,
        match="Model registry functionality is unavailable",
    ):
        tracking_store.get_store("unknown-scheme://")


def test_resolve_tracking_uri_with_param():
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri",
        return_value="databricks://tracking_qoeirj",
    ):
        overriding_uri = "databricks://tracking_poiwerow"
        assert _resolve_tracking_uri(overriding_uri) == overriding_uri


def test_resolve_tracking_uri_with_no_param():
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri",
        return_value="databricks://tracking_zlkjdas",
    ):
        assert _resolve_tracking_uri() == "databricks://tracking_zlkjdas"


def test_store_object_can_be_serialized_by_pickle(tmp_path):
    """
    This test ensures a store object generated by `_get_store` can be serialized by pickle
    to prevent issues such as https://github.com/mlflow/mlflow/issues/2954
    """
    pickle.dump(_get_store(f"file:///{tmp_path.joinpath('mlflow')}"), io.BytesIO())
    pickle.dump(_get_store("databricks"), io.BytesIO())
    pickle.dump(_get_store("https://example.com"), io.BytesIO())
    # pickle.dump(_get_store(f"sqlite:///{tmpdir.strpath}/mlflow.db"), io.BytesIO())
    # This throws `AttributeError: Can't pickle local object 'create_engine.<locals>.connect'`


@pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "relative"])
def test_set_tracking_uri_with_path(tmp_path, monkeypatch, absolute):
    monkeypatch.chdir(tmp_path)
    path = Path("foo/bar")
    if absolute:
        path = tmp_path / path
    with mock.patch("mlflow.tracking._tracking_service.utils._tracking_uri", None):
        set_tracking_uri(path)
        assert get_tracking_uri() == path.absolute().resolve().as_uri()


def test_set_tracking_uri_update_trace_provider():
    default_uri = mlflow.get_tracking_uri()
    try:
        assert get_tracer_tracking_uri() != "file:///tmp"

        set_tracking_uri("file:///tmp")
        assert get_tracer_tracking_uri() == "file:///tmp"

        set_tracking_uri("https://foo")
        assert get_tracer_tracking_uri() == "https://foo"
    finally:
        # clean up
        set_tracking_uri(default_uri)


@pytest.mark.parametrize("store_uri", ["databricks-uc", "databricks-uc://profile"])
def test_get_store_raises_on_uc_uri(store_uri):
    set_tracking_uri(store_uri)
    with pytest.raises(
        MlflowException,
        match="Setting the tracking URI to a Unity Catalog backend is not "
        "supported in the current version of the MLflow client",
    ):
        mlflow.tracking.MlflowClient()
