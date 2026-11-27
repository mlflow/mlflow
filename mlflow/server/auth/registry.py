from __future__ import annotations

import inspect
import threading
import warnings

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking.registry import StoreRegistry
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.uri import redact_uri_credentials

_AUTH_STORE_ENTRYPOINT_GROUP = "mlflow.auth_store"
_building_auth_store_lock = threading.RLock()


class UnsupportedAuthStoreURIException(MlflowException):
    """Exception thrown when building an auth store with an unsupported URI."""

    def __init__(self, unsupported_uri, supported_uri_schemes):
        message = (
            "Auth store functionality is unavailable; got unsupported URI "
            f"'{redact_uri_credentials(unsupported_uri)}' for the auth backend. "
            f"Supported URI schemes are: {supported_uri_schemes}."
        )
        super().__init__(message, error_code=INVALID_PARAMETER_VALUE)
        self.supported_uri_schemes = supported_uri_schemes


class AuthStoreRegistry(StoreRegistry):
    """Scheme-based registry for basic-auth store implementations."""

    def __init__(self):
        super().__init__(_AUTH_STORE_ENTRYPOINT_GROUP)

    def register_entrypoints(self):
        """Register auth stores provided by other packages."""
        for entrypoint in get_entry_points(self.group_name):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    f"Failure attempting to register auth store for scheme "
                    f"'{entrypoint.name}': {exc}",
                    stacklevel=2,
                )

    def get_store(self, database_uri: str, read_database_uri: str | None = None):
        """Return an initialized auth store instance for the provided URI."""
        with _building_auth_store_lock:
            try:
                builder = self.get_store_builder(database_uri)
            except MlflowException as exc:
                raise UnsupportedAuthStoreURIException(
                    unsupported_uri=database_uri,
                    supported_uri_schemes=list(self._registry.keys()),
                ) from exc

            sig = inspect.signature(builder)
            if len(sig.parameters) == 0:
                store = builder()
                store.init_db(database_uri, read_db_uri=read_database_uri)
                return store
            if "read_database_uri" in sig.parameters:
                return builder(database_uri, read_database_uri=read_database_uri)
            if "read_db_uri" in sig.parameters:
                return builder(database_uri, read_db_uri=read_database_uri)
            return builder(database_uri)


_auth_store_registry: AuthStoreRegistry | None = None


def _get_sqlalchemy_auth_store(database_uri: str, read_database_uri: str | None = None):
    from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

    store = SqlAlchemyStore()
    store.init_db(database_uri, read_db_uri=read_database_uri)
    return store


def _register_default_auth_stores(registry: AuthStoreRegistry) -> None:
    for scheme in DATABASE_ENGINES:
        registry.register(scheme, _get_sqlalchemy_auth_store)


def _get_auth_store_registry() -> AuthStoreRegistry:
    global _auth_store_registry
    if _auth_store_registry is None:
        with _building_auth_store_lock:
            if _auth_store_registry is None:
                registry = AuthStoreRegistry()
                _register_default_auth_stores(registry)
                registry.register_entrypoints()
                _auth_store_registry = registry
    return _auth_store_registry


def get_auth_store(database_uri: str, read_database_uri: str | None = None):
    """Return an auth store instance for the specified database URI."""
    return _get_auth_store_registry().get_store(
        database_uri=database_uri,
        read_database_uri=read_database_uri,
    )
