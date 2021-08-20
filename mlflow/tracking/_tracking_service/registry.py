from mlflow.tracking.registry import CachingStoreRegistry


class TrackingStoreRegistry(CachingStoreRegistry):
    """Scheme-based registry for tracking store implementations"""

    def __init__(self):
        super().__init__("mlflow.tracking_store")

    def _resolve_store_uri(self, store_uri: str) -> str:
        from mlflow.tracking._tracking_service import utils

        return utils._resolve_tracking_uri(store_uri)
