from six.moves import urllib

from mlflow.store.tracking.file_store import FileStore


class PluginFileStore(FileStore):
    """FileStore provided through entrypoints system"""

    def __init__(self, store_uri=None, artifact_uri=None):
        path = urllib.parse.urlparse(store_uri).path if store_uri else None
        self.is_plugin = True
        super(PluginFileStore, self).__init__(path, artifact_uri)
