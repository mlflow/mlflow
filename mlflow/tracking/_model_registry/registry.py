from mlflow.tracking.registry import CachingStoreRegistry


class ModelRegistryStoreRegistry(CachingStoreRegistry):
    """Scheme-based registry for model registry store implementations"""

    def __init__(self):
        super().__init__("mlflow.model_registry_store")

    def _resolve_store_uri(self, store_uri: str) -> str:
        from mlflow.tracking._model_registry import utils

        return utils._resolve_registry_uri(store_uri)
