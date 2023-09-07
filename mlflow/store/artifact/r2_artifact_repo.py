from urllib.parse import urlparse

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
        )

    def parse_s3_compliant_uri(self, uri):
        # r2 uri format(virtual): r2://<bucket-name>@<account-id>.r2.cloudflarestorage.com/<path>
        parsed = urlparse(uri)
        if parsed.scheme != "r2":
            raise Exception(f"Not an R2 URI: {uri}")

        host = parsed.netloc
        path = parsed.path

        bucket = host.split("@")[0]
        if path.startswith("/"):
            path = path[1:]
        return bucket, path

    def _get_s3_client(self, addressing_style="virtual", s3_endpoint_url=None):
        # setup Cloudflare R2 backend to be endpoint_url, otherwise all s3 requests
        # will go to AWS S3 by default
        s3_endpoint_url = self.convert_r2_uri_to_s3_endpoint_url(self.artifact_uri)
        return super()._get_s3_client(
            addressing_style=addressing_style, s3_endpoint_url=s3_endpoint_url
        )

    @staticmethod
    def convert_r2_uri_to_s3_endpoint_url(r2_uri):
        host = urlparse(r2_uri).netloc
        host_without_bucket = host.split("@")[-1]
        return f"https://{host_without_bucket}"
