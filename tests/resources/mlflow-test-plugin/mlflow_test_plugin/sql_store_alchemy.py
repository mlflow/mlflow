import urllib.parse

from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore


class PluginRegistrySqlAlchemyStore(SqlAlchemyStore):
    def __init__(self, store_uri=None):
        path = urllib.parse.urlparse(store_uri).path if store_uri else None
        self.is_plugin = True
        super().__init__(path)
