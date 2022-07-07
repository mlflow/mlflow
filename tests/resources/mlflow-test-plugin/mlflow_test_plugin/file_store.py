import urllib.parse

from mlflow.store.tracking.file_store import FileStore


class PluginFileStore(FileStore):
    """FileStore provided through entrypoints system"""

    def __init__(self, store_uri=None, artifact_uri=None):
        path = urllib.parse.urlparse(store_uri).path if store_uri else None
        self.is_plugin = True
        super().__init__(path, artifact_uri)

    # Implement search_experiments to suppress pylint abstract-method error
    def search_experiments(self, *args, **kwargs):
        return super().search_experiments(*args, **kwargs)
