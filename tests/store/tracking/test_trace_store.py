import inspect
from unittest import mock

import pytest

from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, PERMISSION_DENIED, RESOURCE_DOES_NOT_EXIST
from mlflow.store.tracing.abstract_store import AbstractTraceStore
from mlflow.store.tracing.registry import (
    TraceStoreRegistry,
    _register_default_trace_stores,
    get_trace_store,
    resolve_trace_store_uri,
)
from mlflow.store.tracing.sqlalchemy_store import SqlAlchemyTraceStore
from mlflow.store.tracing.sqlalchemy_workspace_store import (
    WorkspaceAwareSqlAlchemyTraceStore,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry, _build_tracking_store
from mlflow.tracking._tracking_service.utils import _get_sqlalchemy_store


def test_direct_sql_constructor_leaves_trace_binding_to_factory(db_uri, tmp_path):
    store = SqlAlchemyStore(db_uri, tmp_path.as_uri())
    try:
        assert not hasattr(store.trace_store, "_tracking_store")
    finally:
        store._dispose_engine()


def test_factory_binds_after_subclass_constructor_finishes(db_uri, tmp_path):
    class CustomSqlStore(SqlAlchemyStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert not hasattr(self.trace_store, "_tracking_store")
            assert not hasattr(self._default_trace_store, "_tracking_store")
            self.ready = True

    class CustomTraceStore(AbstractTraceStore):
        def set_tracking_store(self, tracking_store):
            assert tracking_store.ready
            super().set_tracking_store(tracking_store)

    trace_store = CustomTraceStore()
    with mock.patch.object(
        trace_store, "set_tracking_store", wraps=trace_store.set_tracking_store
    ) as bind:
        store = _build_tracking_store(
            CustomSqlStore, db_uri, tmp_path.as_uri(), trace_store=trace_store
        )
    try:
        bind.assert_called_once_with(store)
        assert store._default_trace_store._tracking_store is store
    finally:
        store._dispose_engine()


def test_registry_binds_default_once_and_does_not_cache_failed_binding(
    monkeypatch, db_uri, tmp_path
):
    registry = TrackingStoreRegistry()
    registry.register("sqlite", _get_sqlalchemy_store)
    original_bind = SqlAlchemyTraceStore.set_tracking_store
    calls = []

    def bind(trace_store, tracking_store):
        calls.append(tracking_store)
        if len(calls) == 1:
            raise RuntimeError("binding failed")
        return original_bind(trace_store, tracking_store)

    monkeypatch.delenv("MLFLOW_TRACE_STORE_URI", raising=False)
    with mock.patch.object(
        SqlAlchemyTraceStore, "set_tracking_store", autospec=True, side_effect=bind
    ) as bind_mock:
        with pytest.raises(RuntimeError, match="binding failed"):
            registry.get_store(db_uri, tmp_path.as_uri())
        store = registry.get_store(db_uri, tmp_path.as_uri())
        assert registry.get_store(db_uri, tmp_path.as_uri()) is store
    try:
        assert bind_mock.call_count == 2
        assert calls[0] is not store
        assert calls[1] is store
        assert store.trace_store._tracking_store is store
    finally:
        store._dispose_engine()


def test_factory_binds_non_sql_tracking_plugin_after_construction(monkeypatch):
    class CustomStore(RestStore):
        def __init__(self, store_uri, artifact_uri, trace_store):
            AbstractStore.__init__(self, trace_store=trace_store)
            self.ready = True

        @staticmethod
        def supports_trace_stores():
            return True

    trace_store = AbstractTraceStore()
    trace_registry = TraceStoreRegistry()
    trace_builder = mock.Mock(return_value=trace_store)
    trace_registry.register("traces", trace_builder)
    monkeypatch.setattr("mlflow.store.tracing.registry._trace_store_registry", trace_registry)
    registry = TrackingStoreRegistry()
    registry.register("custom", CustomStore)
    with mock.patch.object(
        trace_store, "set_tracking_store", wraps=trace_store.set_tracking_store
    ) as bind:
        store = registry.get_store("custom://tracking", trace_store_uri="traces://data")
    assert store.ready
    assert trace_store._tracking_store is store
    bind.assert_called_once_with(store)
    trace_builder.assert_called_once_with("traces://data")


@pytest.mark.parametrize(
    ("explicit", "configured", "tracking_uri", "expected"),
    [
        ("test://explicit", "test://environment", "sqlite:///tracking.db", "test://explicit"),
        (None, "test://environment", "sqlite:///tracking.db", "test://environment"),
        (None, None, "sqlite:///tracking.db", "sqlite:///tracking.db"),
        (None, "", "https://tracking.example.com", "https://tracking.example.com"),
        (None, None, "file:///tracking", "file:///tracking"),
        ("", "test://environment", "sqlite:///tracking.db", ""),
    ],
)
def test_trace_store_uri_resolution_matches_workspace_precedence(
    monkeypatch, explicit, configured, tracking_uri, expected
):
    if configured is None:
        monkeypatch.delenv("MLFLOW_TRACE_STORE_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACE_STORE_URI", configured)
    assert resolve_trace_store_uri(explicit, tracking_uri) == expected


def test_trace_store_uri_resolution_uses_current_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACE_STORE_URI", raising=False)
    monkeypatch.setattr(
        "mlflow.tracking._tracking_service.utils._tracking_uri", "https://tracking.example.com"
    )
    assert resolve_trace_store_uri() == "https://tracking.example.com"


@pytest.mark.parametrize("workspace_aware", [False, True])
@pytest.mark.parametrize(
    "store_uri",
    [
        "sqlite:///:memory:",
        "postgresql+psycopg2://host/tracking",
        "mysql+pymysql://host/tracking",
        "mssql+pyodbc://host/tracking",
    ],
)
def test_registry_constructs_builtin_sql_trace_stores_by_uri(
    monkeypatch, workspace_aware, store_uri
):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(workspace_aware).lower())
    store = get_trace_store(store_uri)
    expected_class = WorkspaceAwareSqlAlchemyTraceStore if workspace_aware else SqlAlchemyTraceStore
    assert type(store) is expected_class
    assert store.db_uri == store_uri
    assert not hasattr(store, "_tracking_store")


@pytest.mark.parametrize(
    ("tracking_uri", "expected_class"),
    [
        ("https://tracking.example.com", RestStore),
        ("http://tracking.example.com", RestStore),
        ("databricks", DatabricksTracingRestStore),
        ("databricks://profile", DatabricksTracingRestStore),
    ],
)
def test_registry_defaults_to_tracking_uri_for_native_clients(
    monkeypatch, tracking_uri, expected_class
):
    monkeypatch.delenv("MLFLOW_TRACE_STORE_URI", raising=False)
    assert type(get_trace_store(tracking_uri=tracking_uri)) is expected_class


def test_registry_defaults_to_file_tracking_uri(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACE_STORE_URI", raising=False)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    assert isinstance(get_trace_store(tracking_uri=tmp_path.as_uri()), FileStore)


@pytest.mark.parametrize("supports_trace_stores", [False, True])
def test_sql_tracking_default_is_selected_by_registry(
    monkeypatch, db_uri, tmp_path, supports_trace_stores
):
    registry = TraceStoreRegistry()
    trace_store = AbstractTraceStore()
    builder = mock.Mock(return_value=trace_store)
    registry.register("sqlite", builder)
    monkeypatch.setattr("mlflow.store.tracing.registry._trace_store_registry", registry)
    monkeypatch.setattr(
        SqlAlchemyStore, "supports_trace_stores", staticmethod(lambda: supports_trace_stores)
    )
    if not supports_trace_stores:
        with pytest.raises(MlflowException, match="does not support trace stores"):
            _build_tracking_store(SqlAlchemyStore, db_uri, tmp_path.as_uri())
        builder.assert_called_once_with(db_uri)
        assert not hasattr(trace_store, "_tracking_store")
        return
    store = _build_tracking_store(SqlAlchemyStore, db_uri, tmp_path.as_uri())
    try:
        assert store.trace_store is trace_store
        assert trace_store._tracking_store is store
        builder.assert_called_once_with(db_uri)
        with store.use_default_trace_store():
            assert isinstance(store.trace_store, SqlAlchemyTraceStore)
            assert store.trace_store._tracking_store is store
    finally:
        store._dispose_engine()


def test_sql_trace_store_rejects_a_different_database(db_uri, tmp_path):
    store = _build_tracking_store(SqlAlchemyStore, db_uri, tmp_path.as_uri())
    trace_store = SqlAlchemyTraceStore("sqlite:///other.db")
    try:
        with pytest.raises(MlflowException, match="same database URI"):
            trace_store.set_tracking_store(store)
        assert not hasattr(trace_store, "_tracking_store")
    finally:
        store._dispose_engine()


def test_sql_trace_store_rejects_mismatched_workspace_mode(db_uri, tmp_path):
    store = _build_tracking_store(SqlAlchemyStore, db_uri, tmp_path.as_uri())
    trace_store = WorkspaceAwareSqlAlchemyTraceStore(db_uri)
    try:
        with pytest.raises(MlflowException, match="same workspace mode"):
            trace_store.set_tracking_store(store)
        assert not hasattr(trace_store, "_tracking_store")
    finally:
        store._dispose_engine()


def test_tracking_registry_resolves_trace_defaults_before_caching(monkeypatch, db_uri, tmp_path):
    trace_registry = TraceStoreRegistry()
    _register_default_trace_stores(trace_registry)
    builder = mock.Mock(side_effect=lambda uri: AbstractTraceStore())
    trace_registry.register("test", builder)
    monkeypatch.setattr("mlflow.store.tracing.registry._trace_store_registry", trace_registry)
    registry = TrackingStoreRegistry()
    registry.register("sqlite", _get_sqlalchemy_store)
    monkeypatch.setenv("MLFLOW_TRACE_STORE_URI", "test://first")
    first = registry.get_store(db_uri, tmp_path.as_uri())
    monkeypatch.setenv("MLFLOW_TRACE_STORE_URI", "test://second")
    second = registry.get_store(db_uri, tmp_path.as_uri())
    native = registry.get_store(db_uri, tmp_path.as_uri(), trace_store_uri=db_uri)
    try:
        assert first is not second
        assert first.trace_store is not second.trace_store
        assert type(native.trace_store) is SqlAlchemyTraceStore
        assert native.trace_store.db_uri == db_uri
        monkeypatch.delenv("MLFLOW_TRACE_STORE_URI")
        assert registry.get_store(db_uri, tmp_path.as_uri()) is native
        assert builder.call_args_list == [mock.call("test://first"), mock.call("test://second")]
    finally:
        first._dispose_engine()
        second._dispose_engine()
        native._dispose_engine()


BACKEND_OPERATIONS = [
    name
    for name, value in vars(AbstractTraceStore).items()
    if not name.startswith("_")
    and inspect.isfunction(value)
    and name not in ("set_tracking_store", "on_traces_deleted", "log_spans_async")
]


@pytest.mark.parametrize("name", BACKEND_OPERATIONS)
def test_unsupported_operations_raise_mlflow_exception(name):
    method = getattr(AbstractTraceStore(), name)
    arguments = {key: getattr(mock.sentinel, key) for key in inspect.signature(method).parameters}
    with pytest.raises(
        MlflowNotImplementedException, match="^$|AbstractTraceStore|Legacy V2"
    ) as error:
        method(**arguments)
    assert error.value.error_code == "NOT_IMPLEMENTED"


def test_store_backend_support_is_explicit():
    assert AbstractStore.supports_trace_stores() is False
    assert FileStore.supports_trace_stores() is False
    assert RestStore.supports_trace_stores() is False
    assert SqlAlchemyStore.supports_trace_stores() is True
    for store_class in (FileStore, RestStore, SqlAlchemyStore):
        assert not inspect.isabstract(store_class)


def test_sql_subclass_opt_out_is_checked_before_database_initialization():
    class UnsupportedSqlStore(SqlAlchemyStore):
        @staticmethod
        def supports_trace_stores():
            return False

    with pytest.raises(MlflowException, match="does not support trace stores"):
        UnsupportedSqlStore(
            "not-a-database-uri", "unused-artifacts", trace_store=AbstractTraceStore()
        )


def test_backend_is_bound_once_to_one_store():
    backend = AbstractTraceStore()
    first = mock.sentinel.first
    backend.set_tracking_store(first)
    backend.set_tracking_store(first)
    assert backend._tracking_store is first
    with pytest.raises(MlflowException, match="cannot be shared"):
        backend.set_tracking_store(mock.sentinel.second)


def test_default_deleted_trace_hook_cleans_host_references():
    backend = AbstractTraceStore()
    store = mock.create_autospec(SqlAlchemyStore, instance=True)
    backend.set_tracking_store(store)
    backend.on_traces_deleted(["trace"])
    store.on_traces_deleted.assert_called_once_with(["trace"])


def test_default_deleted_trace_hook_logs_failed_cleanup(caplog):
    backend = AbstractTraceStore()
    store = mock.create_autospec(SqlAlchemyStore, instance=True)
    backend.set_tracking_store(store)
    with mock.patch.object(
        store, "on_traces_deleted", side_effect=MlflowException("cleanup unavailable")
    ) as cleanup:
        backend.on_traces_deleted(["trace"])
    cleanup.assert_called_once_with(["trace"])
    assert "cleanup unavailable" in caplog.text


@pytest.mark.parametrize(
    "store_uri",
    [
        "test://host/namespace?option=value%2Fother#fragment",
        "test:///local/path",
        "test:/namespace",
        "TEST://host/namespace",
    ],
)
def test_registry_constructs_backend_with_full_uri_without_tracking_store(store_uri):
    registry = TraceStoreRegistry()
    backend = AbstractTraceStore()
    builder = mock.Mock(return_value=backend)
    registry.register("test", builder)
    assert registry.get_store(store_uri) is backend
    builder.assert_called_once_with(store_uri)
    assert not hasattr(backend, "_tracking_store")


def test_registry_discovers_entrypoints():
    backend = AbstractTraceStore()
    builder = mock.Mock(return_value=backend)
    entrypoint = mock.Mock(**{"load.return_value": builder})
    entrypoint.name = "test"
    with mock.patch(
        "mlflow.tracking.registry.get_entry_points", return_value=[entrypoint]
    ) as entries:
        registry = TraceStoreRegistry()
        registry.register_entrypoints()
    entries.assert_called_once_with("mlflow.trace_stores")
    entrypoint.load.assert_called_once()
    assert registry.get_store("test://host") is backend
    builder.assert_called_once_with("test://host")


def test_registry_rejects_unknown_scheme_without_exposing_uri_credentials():
    with pytest.raises(MlflowException, match="Unknown trace store") as error:
        TraceStoreRegistry().get_store("missing://user:secret@host/path")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("store_uri", ["", "test", "/local/path", "test://["])
def test_registry_rejects_invalid_uri_before_building_store(store_uri):
    registry = TraceStoreRegistry()
    builder = mock.Mock()
    registry.register("test", builder)
    with pytest.raises(MlflowException, match="trace store URI|Trace store URI") as error:
        registry.get_store(store_uri)
    assert error.value.error_code == "INVALID_PARAMETER_VALUE"
    builder.assert_not_called()


def test_registry_rejects_invalid_builder_result():
    registry = TraceStoreRegistry()
    builder = mock.Mock(return_value=object())
    registry.register("test", builder)
    with pytest.raises(MlflowException, match="must return an AbstractTraceStore"):
        registry.get_store("test://host")
    builder.assert_called_once_with("test://host")


def test_tracking_store_cache_distinguishes_full_trace_store_uris(monkeypatch, db_uri, tmp_path):
    trace_registry = TraceStoreRegistry()
    builder = mock.Mock(side_effect=lambda uri: AbstractTraceStore())
    trace_registry.register("test", builder)
    monkeypatch.setattr("mlflow.store.tracing.registry._trace_store_registry", trace_registry)
    registry = TrackingStoreRegistry()
    registry.register("sqlite", _get_sqlalchemy_store)
    first_uri = "test://host/first?option=one"
    second_uri = "test://host/first?option=two"
    first = registry.get_store(db_uri, tmp_path.as_uri(), trace_store_uri=first_uri)
    second = registry.get_store(db_uri, tmp_path.as_uri(), trace_store_uri=second_uri)
    try:
        assert registry.get_store(db_uri, tmp_path.as_uri(), trace_store_uri=first_uri) is first
        assert first is not second
        assert first.trace_store is not second.trace_store
        assert builder.call_args_list == [mock.call(first_uri), mock.call(second_uri)]
    finally:
        first._dispose_engine()
        second._dispose_engine()


@pytest.mark.parametrize(
    ("experiment_ids", "expected"),
    [(None, ["tr-2", "tr-1"]), (["1"], ["tr-1"]), (["2"], ["tr-2"]), ([], [])],
)
def test_default_batch_metadata_uses_point_reads_and_preserves_scope(experiment_ids, expected):
    backend = AbstractTraceStore()
    infos = {
        f"tr-{experiment_id}": TraceInfo(
            trace_id=f"tr-{experiment_id}",
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=1,
            state=TraceState.OK,
        )
        for experiment_id in ("1", "2")
    }

    def get_info(trace_id):
        if trace_id not in infos:
            raise MlflowException("Missing trace", RESOURCE_DOES_NOT_EXIST)
        return infos[trace_id]

    with mock.patch.object(backend, "get_trace_info", side_effect=get_info) as get:
        result = backend.batch_get_trace_infos(
            ["tr-2", "missing", "tr-1", "tr-2"], experiment_ids=experiment_ids
        )
    assert [info.trace_id for info in result] == expected
    if experiment_ids == []:
        get.assert_not_called()
    else:
        assert get.call_args_list == [mock.call("tr-2"), mock.call("missing"), mock.call("tr-1")]


@pytest.mark.parametrize(
    "error",
    [
        MlflowException("Forbidden", PERMISSION_DENIED),
        MlflowException("Unavailable", INTERNAL_ERROR),
        MlflowNotImplementedException("point reads unsupported"),
        NotImplementedError("point reads unsupported"),
    ],
)
def test_default_batch_metadata_propagates_non_missing_errors(error):
    backend = AbstractTraceStore()
    with mock.patch.object(backend, "get_trace_info", side_effect=error) as get:
        with pytest.raises(type(error), match=error.args[0]) as result:
            backend.batch_get_trace_infos(["trace"])
    assert result.value is error
    get.assert_called_once_with("trace")


def test_default_batch_metadata_rejects_locations_without_point_reads():
    backend = AbstractTraceStore()
    with mock.patch.object(backend, "get_trace_info") as get:
        with pytest.raises(MlflowNotImplementedException, match="does not support location"):
            backend.batch_get_trace_infos(["trace"], location="catalog.schema")
    get.assert_not_called()


def test_default_batch_metadata_empty_ids_does_not_read():
    backend = AbstractTraceStore()
    with mock.patch.object(backend, "get_trace_info") as get:
        assert backend.batch_get_trace_infos([]) == []
    get.assert_not_called()
