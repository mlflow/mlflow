import inspect
import threading
from functools import lru_cache

from mlflow.tracking.registry import StoreRegistry

_building_store_lock = threading.Lock()


class ModelRegistryStoreRegistry(StoreRegistry):
    """Scheme-based registry for model registry store implementations

    This class allows the registration of a function or class to provide an
    implementation for a given scheme of `store_uri` through the `register`
    methods. Implementations declared though the entrypoints
    `mlflow.registry_store` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating a store through the `get_store` method, the scheme of
    the store URI provided (or inferred from environment) will be used to
    select which implementation to instantiate, which will be called with same
    arguments passed to the `get_store` method.
    """

    def __init__(self):
        super().__init__("mlflow.model_registry_store")

    def get_store(self, store_uri=None, tracking_uri=None):
        """Get a store from the registry based on the scheme of store_uri

        Args:
            store_uri: The store URI. If None, it will be inferred from the environment. This URI
                is used to select which tracking store implementation to instantiate and
                is passed to the constructor of the implementation.
            tracking_uri: The optional string tracking URI to use for any MLflow tracking-related
                operations in the registry client, e.g. downloading source run
                artifacts in order to re-upload them to the model registry location.

        Returns:
            An instance of `mlflow.store.model_registry.AbstractStore` that fulfills the
            store URI requirements.

        """
        from mlflow.tracking._model_registry.utils import _resolve_registry_uri
        from mlflow.tracking._tracking_service.utils import _resolve_tracking_uri

        resolved_store_uri = _resolve_registry_uri(store_uri)
        resolved_tracking_uri = _resolve_tracking_uri(tracking_uri)
        return self._get_store_with_resolved_uri(resolved_store_uri, resolved_tracking_uri)

    @lru_cache(maxsize=100)
    def _get_store_with_resolved_uri(self, resolved_store_uri, resolved_tracking_uri):
        """
        Retrieve the store associated with a resolved (non-None) store URI and an artifact URI.
        Caching is done on resolved URIs because the meaning of an unresolved (None) URI may change
        depending on external configuration, such as environment variables
        """
        with _building_store_lock:
            builder = self.get_store_builder(resolved_store_uri)
            builder_param_names = set(inspect.signature(builder).parameters.keys())
            if "store_uri" in builder_param_names and "tracking_uri" in builder_param_names:
                return builder(store_uri=resolved_store_uri, tracking_uri=resolved_tracking_uri)
            else:
                # Not all model registry stores accept a tracking_uri parameter
                # (e.g. old plugins may not recognize it), so we fall back to
                # passing just the registry URI
                return builder(store_uri=resolved_store_uri)
