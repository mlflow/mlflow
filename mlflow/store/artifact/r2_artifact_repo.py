import urllib.parse

from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository


class R2ArtifactRepository(S3ArtifactRepository):
    """Stores artifacts on Cloudflare R2."""

    def __init__(
        self, artifact_uri, access_key_id=None, secret_access_key=None, session_token=None
    ):
        super().__init__(
            artifact_uri,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            addressing_style="virtual",
        )

    def parse_s3_compliant_uri(self, uri):
        # r2 uri format(virtual): r2://<bucket-name>@<account-id>.r2.cloudflarestorage.com/<path>
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "r2":
            raise Exception(f"Not an R2 URI: {uri}")

        host = parsed.netloc
        path = parsed.path

        bucket = host.split("@")[0]
        if path.startswith("/"):
            path = path[1:]
        return bucket, path
