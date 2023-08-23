from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository

class R2ArtifactRepository(S3ArtifactRepository):
    """Stores artifacts on Cloudflare R2."""

    def __init__(self, artifact_uri, access_key_id=None, secret_access_key=None, session_token=None):
        super.__init__(access_key_id, secret_access_key, session_token)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token

