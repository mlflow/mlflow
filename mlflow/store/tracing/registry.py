from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES, MLFLOW_TRACE_STORE_URI
from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.store.tracing.abstract_store import AbstractTraceStore
from mlflow.tracking.registry import StoreRegistry
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.uri import get_uri_scheme

if TYPE_CHECKING:
    from mlflow.store.tracing.sqlalchemy_store import SqlAlchemyTraceStore

TraceStoreBuilder = Callable[[str], AbstractTraceStore]


class TraceStoreRegistry(StoreRegistry):
    """Discover URI-scheme builders from ``mlflow.trace_stores`` entry points."""

    def __init__(self):
        super().__init__("mlflow.trace_stores")

    def register(self, scheme: str, store_builder: TraceStoreBuilder) -> None:
        super().register(scheme, store_builder)

    def get_store(
        self, store_uri: str, *, workspace_aware: bool | None = None
    ) -> AbstractTraceStore:
        try:
            scheme = "databricks" if store_uri == "databricks" else get_uri_scheme(store_uri)
        except ValueError as e:
            raise MlflowException.invalid_parameter_value("Invalid trace store URI.") from e
        if not store_uri or (not scheme and "" not in self._registry):
            raise MlflowException.invalid_parameter_value(
                "Trace store URI must include a registered scheme, for example "
                "'my-traces://host/path'."
            )
        if scheme not in self._registry:
            raise MlflowException.invalid_parameter_value(
                f"Unknown trace store scheme {scheme!r}. "
                f"Registered schemes: {sorted(self._registry)}. "
                "Install a provider registering an 'mlflow.trace_stores' entry point."
            )
        builder = self._registry[scheme]
        store = (
            _get_sqlalchemy_trace_store(store_uri, workspace_aware=workspace_aware)
            if builder is _get_sqlalchemy_trace_store
            else builder(store_uri)
        )
        if not isinstance(store, AbstractTraceStore):
            raise MlflowException.invalid_parameter_value(
                f"Trace store scheme {scheme!r} must return an AbstractTraceStore instance; "
                f"received {type(store).__name__}."
            )
        return store


def resolve_trace_store_uri(
    trace_store_uri: str | None = None, tracking_uri: str | None = None
) -> str:
    """Resolve an explicit trace URI, then the environment override, then the tracking URI."""
    if trace_store_uri is not None:
        return trace_store_uri
    if configured_uri := MLFLOW_TRACE_STORE_URI.get():
        return configured_uri

    from mlflow.tracking._tracking_service.utils import _resolve_tracking_uri

    return _resolve_tracking_uri(tracking_uri)


def _get_sqlalchemy_trace_store(
    store_uri: str, *, workspace_aware: bool | None = None
) -> SqlAlchemyTraceStore:
    from mlflow.store.tracing.sqlalchemy_store import SqlAlchemyTraceStore
    from mlflow.store.tracing.sqlalchemy_workspace_store import (
        WorkspaceAwareSqlAlchemyTraceStore,
    )

    if workspace_aware is None:
        workspace_aware = MLFLOW_ENABLE_WORKSPACES.get()
    store_class = WorkspaceAwareSqlAlchemyTraceStore if workspace_aware else SqlAlchemyTraceStore
    return store_class(store_uri)


def _get_file_trace_store(store_uri: str) -> AbstractTraceStore:
    from mlflow.store.tracking.file_store import FileStore

    return FileStore(store_uri, store_uri)


def _get_rest_trace_store(store_uri: str) -> AbstractTraceStore:
    from mlflow.store.tracking.rest_store import RestStore

    return RestStore(partial(get_default_host_creds, store_uri))


def _get_databricks_trace_store(store_uri: str) -> AbstractTraceStore:
    from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore

    return DatabricksTracingRestStore(partial(get_databricks_host_creds, store_uri))


def _register_default_trace_stores(registry: TraceStoreRegistry) -> None:
    for scheme in DATABASE_ENGINES:
        registry.register(scheme, _get_sqlalchemy_trace_store)
    for scheme in ("", "file"):
        registry.register(scheme, _get_file_trace_store)
    for scheme in ("http", "https"):
        registry.register(scheme, _get_rest_trace_store)
    registry.register("databricks", _get_databricks_trace_store)


_trace_store_registry: TraceStoreRegistry | None = None
_building_trace_store_registry_lock = threading.RLock()


def _get_trace_store_registry() -> TraceStoreRegistry:
    global _trace_store_registry
    if _trace_store_registry is None:
        with _building_trace_store_registry_lock:
            if _trace_store_registry is None:
                registry = TraceStoreRegistry()
                _register_default_trace_stores(registry)
                registry.register_entrypoints()
                _trace_store_registry = registry
    return _trace_store_registry


def get_trace_store(
    trace_store_uri: str | None = None,
    tracking_uri: str | None = None,
    *,
    workspace_aware: bool | None = None,
) -> AbstractTraceStore:
    return _get_trace_store_registry().get_store(
        resolve_trace_store_uri(trace_store_uri, tracking_uri),
        workspace_aware=workspace_aware,
    )
