from typing import Any, Callable

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.abstract_trace_store import AbstractTraceStore
from mlflow.tracking.registry import StoreRegistry

TraceStoreBuilder = Callable[[dict[str, Any]], AbstractTraceStore]


class TraceStoreRegistry(StoreRegistry):
    """Discover independent backend builders from ``mlflow.trace_stores`` entry points."""

    def __init__(self):
        super().__init__("mlflow.trace_stores")

    def register(self, kind: str, store_builder: TraceStoreBuilder) -> None:
        super().register(kind, store_builder)

    def get_store(
        self, kind: str, store_config: dict[str, Any] | None = None
    ) -> AbstractTraceStore:
        if kind not in self._registry:
            raise MlflowException.invalid_parameter_value(
                f"Unknown trace store {kind!r}. Registered backends: {sorted(self._registry)}. "
                "Install a provider registering an 'mlflow.trace_stores' entry point."
            )
        backend = self._registry[kind]({} if store_config is None else store_config)
        if not isinstance(backend, AbstractTraceStore):
            raise MlflowException.invalid_parameter_value(
                f"Trace store {kind!r} must return an AbstractTraceStore instance; "
                f"received {type(backend).__name__}."
            )
        return backend


_trace_store_registry = TraceStoreRegistry()
_trace_store_registry.register_entrypoints()
