from functools import lru_cache

from mlflow.tracking.registry import StoreRegistry


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

    def get_store(self, store_uri=None):
        """Get a store from the registry based on the scheme of store_uri

        :param store_uri: The store URI. If None, it will be inferred from the environment. This URI
                          is used to select which tracking store implementation to instantiate and
                          is passed to the constructor of the implementation.

        :return: An instance of `mlflow.store.model_registry.AbstractStore` that fulfills the
                 store URI requirements.
        """
        from mlflow.tracking._model_registry import utils

        resolved_store_uri = utils._resolve_registry_uri(store_uri)
        return self._get_store_with_resolved_uri(resolved_store_uri)

    @lru_cache(maxsize=100)
    def _get_store_with_resolved_uri(self, resolved_store_uri):
        """
        Retrieve the store associated with a resolved (non-None) store URI and an artifact URI.
        Caching is done on resolved URIs because the meaning of an unresolved (None) URI may change
        depending on external configuration, such as environment variables
        """
        builder = self.get_store_builder(resolved_store_uri)
        return builder(store_uri=resolved_store_uri)
