from __future__ import annotations

import inspect
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from mlflow.tracking.registry import StoreRegistry

if TYPE_CHECKING:
    from mlflow.store.tracking.abstract_store import AbstractStore

_P = ParamSpec("_P")
_Store = TypeVar("_Store", bound="AbstractStore")
_building_store_lock = threading.Lock()


def _build_tracking_store(
    builder: Callable[_P, _Store], *args: _P.args, **kwargs: _P.kwargs
) -> _Store:
    """Construct the entire tracking store before binding its trace stores."""
    from mlflow.store.tracking.abstract_store import AbstractStore

    store = builder(*args, **kwargs)
    if isinstance(store, AbstractStore):
        for trace_store in store._trace_stores:
            trace_store.set_tracking_store(store)
    return store


class TrackingStoreRegistry(StoreRegistry):
    """Scheme-based registry for tracking store implementations

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints
    `mlflow.tracking_store` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    def __init__(self):
        super().__init__("mlflow.tracking_store")

    def get_store(self, store_uri=None, artifact_uri=None, *, trace_store_uri=None):
        """Get a store from the registry based on the scheme of store_uri

        Args:
            store_uri: The store URI. If None, it will be inferred from the environment. This URI
                is used to select which tracking store implementation to instantiate and
                is passed to the constructor of the implementation.
            artifact_uri: Artifact repository URI. Passed through to the tracking store
                implementation.
            trace_store_uri: Explicit trace store URI. Otherwise, use MLFLOW_TRACE_STORE_URI
                or the resolved tracking URI. Its scheme selects the trace implementation.
                Independent trace stores are passed to the tracking-store builder through the
                trace_store constructor keyword.

        Returns:
            An instance of `mlflow.store.tracking.AbstractStore` that fulfills the store URI
            requirements.

        """
        from mlflow.store.tracing.registry import resolve_trace_store_uri
        from mlflow.tracking._tracking_service import utils

        resolved_store_uri = utils._resolve_tracking_uri(store_uri)
        resolved_trace_store_uri = resolve_trace_store_uri(trace_store_uri, resolved_store_uri)
        return self._get_store_with_resolved_uri(
            resolved_store_uri, artifact_uri, resolved_trace_store_uri
        )

    @lru_cache(maxsize=100)
    def _get_store_with_resolved_uri(
        self, resolved_store_uri, artifact_uri=None, trace_store_uri=None
    ):
        """
        Retrieve the store associated with a resolved (non-None) store URI and an artifact URI.
        Caching is done on resolved URIs because the meaning of an unresolved (None) URI may change
        depending on external configuration, such as environment variables
        """
        with _building_store_lock:
            builder = self.get_store_builder(resolved_store_uri)
            if trace_store_uri is not None and trace_store_uri != resolved_store_uri:
                # Imported only for explicit server/admin selection to avoid client import cycles.
                from mlflow.exceptions import MlflowException
                from mlflow.store.tracing.registry import _get_trace_store_registry

                try:
                    inspect.signature(builder).bind(
                        store_uri=resolved_store_uri,
                        artifact_uri=artifact_uri,
                        trace_store=None,
                    )
                except TypeError as e:
                    raise MlflowException.invalid_parameter_value(
                        "The selected tracking-store builder does not accept a trace_store "
                        "constructor argument."
                    ) from e
                backend = _get_trace_store_registry().get_store(trace_store_uri)
                store = _build_tracking_store(
                    builder,
                    store_uri=resolved_store_uri,
                    artifact_uri=artifact_uri,
                    trace_store=backend,
                )
                if not store.supports_trace_stores():
                    raise MlflowException.invalid_parameter_value(
                        "The selected tracking store does not support trace stores."
                    )
                return store
            return _build_tracking_store(
                builder, store_uri=resolved_store_uri, artifact_uri=artifact_uri
            )
