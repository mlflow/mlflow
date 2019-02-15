from six.moves import urllib

from mlflow.store.file_store import FileStore
from mlflow.store.local_artifact_repo import LocalArtifactRepository


class PluginFileStore(FileStore):
    """FileStore provided through entrypoints system"""


def get_file_store(store_uri, **_):
    """Return instance of PluginFileStore"""
    path = urllib.parse.urlparse(store_uri).path if store_uri else None
    return PluginFileStore(path, path)


class PluginLocalArtifactRepository(LocalArtifactRepository):
    """LocalArtifactRepository provided through plugin system"""
