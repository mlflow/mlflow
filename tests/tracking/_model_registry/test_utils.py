import io
import pickle
from unittest import mock

import pytest

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._model_registry.utils import (
    _get_store,
    _resolve_registry_uri,
    get_registry_uri,
    set_registry_uri,
)
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException

# Disable mocking tracking URI here, as we want to test setting the tracking URI via
# environment variable. See
# http://doc.pytest.org/en/latest/skipping.html#skip-all-test-functions-of-a-class-or-module
# and https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#writing-python-tests
# for more information.
pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def reset_registry_uri():
    yield
    set_registry_uri(None)


def test_set_get_registry_uri():
    with mock.patch(
        "mlflow.tracking._model_registry.utils._resolve_tracking_uri"
    ) as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = "databricks://tracking_sldkfj"
        uri = "databricks://registry/path"
        set_registry_uri(uri)
        assert get_registry_uri() == uri
        set_registry_uri(None)


def test_set_get_empty_registry_uri():
    with mock.patch(
        "mlflow.tracking._model_registry.utils._resolve_tracking_uri"
    ) as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = None
        set_registry_uri("")
        assert get_registry_uri() is None
        set_registry_uri(None)


def test_default_get_registry_uri_no_tracking_uri():
    with mock.patch(
        "mlflow.tracking._model_registry.utils._resolve_tracking_uri"
    ) as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = None
        set_registry_uri(None)
        assert get_registry_uri() is None


def test_default_get_registry_uri_with_databricks_tracking_uri_defaults_to_uc():
    """Test that databricks tracking URIs default to databricks-uc for registry"""
    tracking_uri = "databricks://tracking_werohoz"
    with mock.patch(
        "mlflow.tracking._model_registry.utils._resolve_tracking_uri"
    ) as resolve_tracking_uri_mock:
        resolve_tracking_uri_mock.return_value = tracking_uri
        set_registry_uri(None)
        # Should default to Unity Catalog when tracking URI starts with 'databricks'
        # and include the profile when present
        assert get_registry_uri() == "databricks-uc://tracking_werohoz"


@pytest.mark.parametrize(
    "tracking_uri",
    [
        "http://localhost:5000",
        "https://remote-server.com",
        "sqlite:///path/to/db.sqlite",
        "postgresql://user:pass@localhost/db",
        "file:///local/path",
    ],
)
def test_default_registry_uri_non_databricks_tracking_uri(tracking_uri):
    """Test that non-databricks tracking URIs are used directly as registry URI"""
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri"
    ) as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tracking_uri
        set_registry_uri(None)
        # Non-databricks URIs should be used directly as registry URI
        assert get_registry_uri() == tracking_uri


@pytest.mark.parametrize(
    ("tracking_uri", "expected_registry_uri"),
    [
        ("databricks", "databricks-uc"),
        ("databricks://profile", "databricks-uc://profile"),
        ("databricks://profile_name", "databricks-uc://profile_name"),
        ("databricks://workspace_url", "databricks-uc://workspace_url"),
        (
            "databricks://some.databricks.workspace.com",
            "databricks-uc://some.databricks.workspace.com",
        ),
    ],
)
def test_databricks_tracking_uri_variations_default_to_uc(tracking_uri, expected_registry_uri):
    """Test that various databricks tracking URI formats default to databricks-uc"""
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri"
    ) as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tracking_uri
        set_registry_uri(None)
        # All databricks tracking URIs should default to Unity Catalog
        registry_uri = get_registry_uri()
        assert registry_uri == expected_registry_uri


def test_explicit_registry_uri_overrides_databricks_default():
    """Test that explicitly set registry URI takes precedence over databricks default"""
    tracking_uri = "databricks://workspace"
    explicit_registry_uri = "databricks://different_workspace"

    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri"
    ) as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tracking_uri
        set_registry_uri(explicit_registry_uri)
        # Explicit registry URI should override the databricks-uc default
        assert get_registry_uri() == explicit_registry_uri
        set_registry_uri(None)  # Reset for other tests


def test_registry_uri_from_environment_overrides_databricks_default():
    """Test that registry URI from environment variable overrides databricks default"""
    from mlflow.environment_variables import MLFLOW_REGISTRY_URI

    tracking_uri = "databricks://workspace"
    env_registry_uri = "http://env-registry-server:5000"

    with (
        mock.patch(
            "mlflow.tracking._tracking_service.utils.get_tracking_uri"
        ) as get_tracking_uri_mock,
        mock.patch.object(MLFLOW_REGISTRY_URI, "get", return_value=env_registry_uri),
    ):
        get_tracking_uri_mock.return_value = tracking_uri
        set_registry_uri(None)
        # Environment variable should override the databricks-uc default
        assert get_registry_uri() == env_registry_uri


def test_registry_uri_from_spark_session_overrides_databricks_default():
    """Test that registry URI from Spark session overrides databricks default"""
    tracking_uri = "databricks://workspace"
    spark_registry_uri = "databricks-uc://spark_profile"

    with (
        mock.patch(
            "mlflow.tracking._tracking_service.utils.get_tracking_uri"
        ) as get_tracking_uri_mock,
        mock.patch(
            "mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session"
        ) as get_spark_uri_mock,
    ):
        get_tracking_uri_mock.return_value = tracking_uri
        get_spark_uri_mock.return_value = spark_registry_uri
        set_registry_uri(None)
        # Spark session URI should override the databricks-uc default
        assert get_registry_uri() == spark_registry_uri


@pytest.mark.parametrize(
    ("tracking_uri", "expected_result"),
    [
        ("mydatabricks://custom", None),  # Should not match partial
        ("databricks", "databricks-uc"),  # Should match exact
        ("", None),  # Empty string should return None
    ],
)
def test_edge_cases_for_databricks_uri_detection(tracking_uri, expected_result):
    """Test edge cases for databricks URI detection"""
    with mock.patch(
        "mlflow.tracking._tracking_service.utils.get_tracking_uri"
    ) as get_tracking_uri_mock:
        get_tracking_uri_mock.return_value = tracking_uri
        set_registry_uri(None)
        result = get_registry_uri()
        if expected_result is None:
            assert result == tracking_uri  # Should fallback to tracking URI
        else:
            assert result == expected_result


@pytest.mark.parametrize(
    ("tracking_uri", "registry_uri_param", "expected_result"),
    [
        # (tracking_uri, registry_uri_param, expected_result)
        ("databricks://workspace", None, "databricks-uc://workspace"),
        ("databricks", None, "databricks-uc"),
        ("http://localhost:5000", None, "http://localhost:5000"),
        ("databricks://workspace", "explicit://registry", "explicit://registry"),
        (None, None, None),
        ("", None, ""),
    ],
)
def test_resolve_registry_uri_consistency_with_get_registry_uri(
    tracking_uri, registry_uri_param, expected_result
):
    """Test that _resolve_registry_uri behaves consistently with get_registry_uri"""
    with mock.patch(
        "mlflow.tracking._model_registry.utils._resolve_tracking_uri"
    ) as mock_resolve_tracking:
        mock_resolve_tracking.return_value = tracking_uri
        set_registry_uri(None)  # Clear context

        result = _resolve_registry_uri(registry_uri_param, tracking_uri)
        assert result == expected_result, (
            f"Failed for tracking_uri={tracking_uri}, registry_uri={registry_uri_param}"
        )


def test_resolve_registry_uri_with_environment_variable():
    """Test _resolve_registry_uri with environment variable set"""
    from mlflow.environment_variables import MLFLOW_REGISTRY_URI

    env_registry_uri = "http://env-registry:5000"
    tracking_uri = "databricks://workspace"

    with (
        mock.patch(
            "mlflow.tracking._model_registry.utils._resolve_tracking_uri"
        ) as mock_resolve_tracking,
        mock.patch.object(MLFLOW_REGISTRY_URI, "get", return_value=env_registry_uri),
    ):
        mock_resolve_tracking.return_value = tracking_uri
        set_registry_uri(None)  # Clear explicit setting

        # Environment variable should override databricks default
        result = _resolve_registry_uri(None, tracking_uri)
        assert result == env_registry_uri


def test_resolve_registry_uri_with_spark_session():
    """Test _resolve_registry_uri with Spark session URI"""
    spark_registry_uri = "databricks-uc://spark_profile"
    tracking_uri = "databricks://workspace"

    with (
        mock.patch(
            "mlflow.tracking._model_registry.utils._resolve_tracking_uri"
        ) as mock_resolve_tracking,
        mock.patch(
            "mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session"
        ) as mock_spark_uri,
    ):
        mock_resolve_tracking.return_value = tracking_uri
        mock_spark_uri.return_value = spark_registry_uri
        set_registry_uri(None)  # Clear explicit setting

        # Spark session URI should override databricks default
        result = _resolve_registry_uri(None, tracking_uri)
        assert result == spark_registry_uri


def test_get_store_rest_store_from_arg(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "https://my-tracking-server:5050")
    store = _get_store("http://some/path")
    assert isinstance(store, RestStore)
    assert store.get_host_creds().host == "http://some/path"


def test_fallback_to_tracking_store(monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "https://my-tracking-server:5050")
    store = _get_store()
    assert isinstance(store, RestStore)
    assert store.get_host_creds().host == "https://my-tracking-server:5050"
    assert store.get_host_creds().token is None


@pytest.mark.parametrize("db_type", DATABASE_ENGINES)
def test_get_store_sqlalchemy_store(db_type, monkeypatch):
    uri = f"{db_type}://hostname/database"
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, uri)
    monkeypatch.delenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", raising=False)
    with (
        mock.patch("sqlalchemy.create_engine") as mock_create_engine,
        mock.patch("mlflow.store.db.utils._initialize_tables"),
        mock.patch(
            "mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore."
            "_verify_registry_tables_exist"
        ),
    ):
        store = _get_store()
        assert isinstance(store, SqlAlchemyStore)
        assert store.db_uri == uri

    mock_create_engine.assert_called_once_with(uri, pool_pre_ping=True)


@pytest.mark.parametrize("bad_uri", ["badsql://imfake", "yoursql://hi"])
def test_get_store_bad_uris(bad_uri, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, bad_uri)
    with pytest.raises(
        UnsupportedModelRegistryStoreURIException,
        match="Model registry functionality is unavailable",
    ):
        _get_store()


def test_get_store_caches_on_store_uri(tmp_path):
    store_uri_1 = f"sqlite:///{tmp_path.joinpath('store1.db')}"
    store_uri_2 = f"sqlite:///{tmp_path.joinpath('store2.db')}"

    store1 = _get_store(store_uri_1)
    store2 = _get_store(store_uri_1)
    assert store1 is store2

    store3 = _get_store(store_uri_2)
    store4 = _get_store(store_uri_2)
    assert store3 is store4

    assert store1 is not store3


@pytest.mark.parametrize("store_uri", ["databricks-uc", "databricks-uc://profile"])
def test_get_store_uc_registry_uri(store_uri):
    assert isinstance(_get_store(store_uri), UcModelRegistryStore)


def test_store_object_can_be_serialized_by_pickle():
    """
    This test ensures a store object generated by `_get_store` can be serialized by pickle
    to prevent issues such as https://github.com/mlflow/mlflow/issues/2954
    """
    pickle.dump(_get_store("https://example.com"), io.BytesIO())
    pickle.dump(_get_store("databricks"), io.BytesIO())
    # pickle.dump(_get_store(f"sqlite:///{tmpdir.strpath}/mlflow.db"), io.BytesIO())
    # This throws `AttributeError: Can't pickle local object 'create_engine.<locals>.connect'`
