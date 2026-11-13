from collections.abc import Callable

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.abstract_trace_store import AbstractTraceStore
from mlflow.tracking.registry import StoreRegistry
from mlflow.utils.uri import get_uri_scheme

TraceStoreBuilder = Callable[[str], AbstractTraceStore]


class TraceStoreRegistry(StoreRegistry):
    """Discover URI-scheme builders from ``mlflow.trace_stores`` entry points."""

    def __init__(self):
        super().__init__("mlflow.trace_stores")

    def register(self, scheme: str, store_builder: TraceStoreBuilder) -> None:
        super().register(scheme, store_builder)

    def get_store(self, store_uri: str) -> AbstractTraceStore:
        try:
            scheme = get_uri_scheme(store_uri)
        except ValueError as e:
            raise MlflowException.invalid_parameter_value("Invalid trace store URI.") from e
        if not scheme:
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
        store = self._registry[scheme](store_uri)
        if not isinstance(store, AbstractTraceStore):
            raise MlflowException.invalid_parameter_value(
                f"Trace store scheme {scheme!r} must return an AbstractTraceStore instance; "
                f"received {type(store).__name__}."
            )
        return store


_trace_store_registry = TraceStoreRegistry()
_trace_store_registry.register_entrypoints()
