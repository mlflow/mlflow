from six.moves import urllib

from mlflow.store.tracking.file_store import FileStore
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.tracking.context.abstract_context import RunContextProvider


class PluginFileStore(FileStore):
    """FileStore provided through entrypoints system"""

    def __init__(self, store_uri=None, artifact_uri=None):
        path = urllib.parse.urlparse(store_uri).path if store_uri else None
        self.is_plugin = True
        super(PluginFileStore, self).__init__(path, artifact_uri)


class PluginLocalArtifactRepository(LocalArtifactRepository):
    """LocalArtifactRepository provided through plugin system"""
    is_plugin = True


class PluginRunContextProvider(RunContextProvider):
    """RunContextProvider provided through plugin system"""

    def in_context(self):
        return False

    def tags(self):
        return {"test": "tag"}
