import urllib.parse

from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore


class PluginJobStore(SqlAlchemyJobStore):
    def __init__(self, store_uri):
        path = urllib.parse.urlparse(store_uri).path
        self.is_plugin = True
        super().__init__(f"sqlite:///{path}")
