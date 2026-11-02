import inspect
from unittest import mock

import pytest

from mlflow.entities import TraceInfo, TraceLocation, TraceState
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, PERMISSION_DENIED, RESOURCE_DOES_NOT_EXIST
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.abstract_trace_backend import AbstractTraceBackend
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.trace_backend_registry import TraceBackendRegistry

BACKEND_OPERATIONS = [
    name
    for name, value in vars(AbstractTraceBackend).items()
    if not name.startswith("_")
    and inspect.isfunction(value)
    and name not in ("set_tracking_store", "on_traces_deleted", "log_spans_async")
]


@pytest.mark.parametrize("name", BACKEND_OPERATIONS)
def test_unsupported_operations_raise_mlflow_exception(name):
    method = getattr(AbstractTraceBackend(), name)
    arguments = {key: getattr(mock.sentinel, key) for key in inspect.signature(method).parameters}
    with pytest.raises(MlflowNotImplementedException, match="^$|AbstractTraceBackend") as error:
        method(**arguments)
    assert error.value.error_code == "NOT_IMPLEMENTED"


def test_store_backend_support_is_explicit():
    assert AbstractStore.supports_trace_backends() is False
    assert FileStore.supports_trace_backends() is False
    assert RestStore.supports_trace_backends() is False
    assert SqlAlchemyStore.supports_trace_backends() is True
    for store_class in (FileStore, RestStore, SqlAlchemyStore):
        assert not inspect.isabstract(store_class)


def test_sql_subclass_opt_out_is_checked_before_database_initialization():
    class UnsupportedSqlStore(SqlAlchemyStore):
        @staticmethod
        def supports_trace_backends():
            return False

    with pytest.raises(MlflowException, match="does not support trace backends"):
        UnsupportedSqlStore(
            "not-a-database-uri", "unused-artifacts", trace_backend=AbstractTraceBackend()
        )


def test_backend_is_bound_once_to_one_store():
    backend = AbstractTraceBackend()
    first = mock.sentinel.first
    backend.set_tracking_store(first)
    backend.set_tracking_store(first)
    assert backend._tracking_store is first
    with pytest.raises(MlflowException, match="cannot be shared"):
        backend.set_tracking_store(mock.sentinel.second)


def test_default_deleted_trace_hook_cleans_host_references():
    backend = AbstractTraceBackend()
    store = mock.create_autospec(SqlAlchemyStore, instance=True)
    backend.set_tracking_store(store)
    backend.on_traces_deleted(["trace"])
    store.on_traces_deleted.assert_called_once_with(["trace"])


def test_default_deleted_trace_hook_logs_failed_cleanup(caplog):
    backend = AbstractTraceBackend()
    store = mock.create_autospec(SqlAlchemyStore, instance=True)
    backend.set_tracking_store(store)
    with mock.patch.object(
        store, "on_traces_deleted", side_effect=MlflowException("cleanup unavailable")
    ) as cleanup:
        backend.on_traces_deleted(["trace"])
    cleanup.assert_called_once_with(["trace"])
    assert "cleanup unavailable" in caplog.text


def test_registry_constructs_backend_without_tracking_store():
    registry = TraceBackendRegistry()
    backend = AbstractTraceBackend()
    builder = mock.Mock(return_value=backend)
    registry.register("test", builder)
    config = {"setting": "value"}
    assert registry.get_backend("test", config) is backend
    builder.assert_called_once_with(config)
    assert not hasattr(backend, "_tracking_store")


def test_registry_discovers_entrypoints():
    backend = AbstractTraceBackend()
    builder = mock.Mock(return_value=backend)
    entrypoint = mock.Mock(**{"load.return_value": builder})
    entrypoint.name = "test"
    with mock.patch(
        "mlflow.tracking.registry.get_entry_points", return_value=[entrypoint]
    ) as entries:
        registry = TraceBackendRegistry()
        registry.register_entrypoints()
    entries.assert_called_once_with("mlflow.trace_backends")
    entrypoint.load.assert_called_once()
    assert registry.get_backend("test") is backend
    builder.assert_called_once_with({})


def test_registry_rejects_unknown_kind():
    with pytest.raises(MlflowException, match="Unknown trace backend"):
        TraceBackendRegistry().get_backend("missing")


def test_registry_rejects_invalid_builder_result():
    registry = TraceBackendRegistry()
    builder = mock.Mock(return_value=object())
    registry.register("test", builder)
    with pytest.raises(MlflowException, match="must return an AbstractTraceBackend"):
        registry.get_backend("test")
    builder.assert_called_once_with({})


@pytest.mark.parametrize(
    ("experiment_ids", "expected"),
    [(None, ["tr-2", "tr-1"]), (["1"], ["tr-1"]), (["2"], ["tr-2"]), ([], [])],
)
def test_default_batch_metadata_uses_point_reads_and_preserves_scope(experiment_ids, expected):
    backend = AbstractTraceBackend()
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
    backend = AbstractTraceBackend()
    with mock.patch.object(backend, "get_trace_info", side_effect=error) as get:
        with pytest.raises(type(error), match=error.args[0]) as result:
            backend.batch_get_trace_infos(["trace"])
    assert result.value is error
    get.assert_called_once_with("trace")


def test_default_batch_metadata_rejects_locations_without_point_reads():
    backend = AbstractTraceBackend()
    with mock.patch.object(backend, "get_trace_info") as get:
        with pytest.raises(MlflowNotImplementedException, match="does not support location"):
            backend.batch_get_trace_infos(["trace"], location="catalog.schema")
    get.assert_not_called()


def test_default_batch_metadata_empty_ids_does_not_read():
    backend = AbstractTraceBackend()
    with mock.patch.object(backend, "get_trace_info") as get:
        assert backend.batch_get_trace_infos([]) == []
    get.assert_not_called()
