from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository


class PluginLocalArtifactRepository(LocalArtifactRepository):
    """LocalArtifactRepository provided through plugin system"""

    is_plugin = True
