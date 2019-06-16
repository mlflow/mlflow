from mlflow.store.local_artifact_repo import LocalArtifactRepository


class DbfsFuseArtifactRepository(LocalArtifactRepository):
    """
    Artifact repository that writes to DBFS using local filesystem APIs via the
    DBFS FUSE mount.
    """
    def __init__(self, artifact_uri):
        file_uri = "file:///dbfs/{}".format(artifact_uri[len("dbfs:/"):])
        super(DbfsFuseArtifactRepository, self).__init__(file_uri)
