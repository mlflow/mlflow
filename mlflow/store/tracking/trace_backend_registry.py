from typing import Any, Callable

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.abstract_trace_backend import AbstractTraceBackend
from mlflow.tracking.registry import StoreRegistry

TraceBackendBuilder = Callable[[dict[str, Any]], AbstractTraceBackend]


class TraceBackendRegistry(StoreRegistry):
    """Discover independent backend builders from ``mlflow.trace_backends`` entry points."""

    def __init__(self):
        super().__init__("mlflow.trace_backends")

    def register(self, kind: str, store_builder: TraceBackendBuilder) -> None:
        super().register(kind, store_builder)

    def get_backend(
        self, kind: str, backend_config: dict[str, Any] | None = None
    ) -> AbstractTraceBackend:
        if kind not in self._registry:
            raise MlflowException.invalid_parameter_value(
                f"Unknown trace backend {kind!r}. Registered backends: {sorted(self._registry)}. "
                "Install a provider registering an 'mlflow.trace_backends' entry point."
            )
        backend = self._registry[kind]({} if backend_config is None else backend_config)
        if not isinstance(backend, AbstractTraceBackend):
            raise MlflowException.invalid_parameter_value(
                f"Trace backend {kind!r} must return an AbstractTraceBackend instance; "
                f"received {type(backend).__name__}."
            )
        return backend


_trace_backend_registry = TraceBackendRegistry()
_trace_backend_registry.register_entrypoints()
