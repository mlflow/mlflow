from urllib.parse import urlparse

from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client


class R2ArtifactRepository(OptimizedS3ArtifactRepository):
    """Stores artifacts on Cloudflare R2."""

    def __init__(
        self,
        artifact_uri,
        access_key_id=None,
        secret_access_key=None,
        session_token=None,
        credential_refresh_def=None,
        s3_upload_extra_args=None,
        tracking_uri=None,
        registry_uri: str | None = None,
    ):
        # setup Cloudflare R2 backend to be endpoint_url, otherwise all s3 requests
        # will go to AWS S3 by default
        s3_endpoint_url = self.convert_r2_uri_to_s3_endpoint_url(artifact_uri)
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._s3_endpoint_url = s3_endpoint_url
        self.bucket, self.bucket_path = self.parse_s3_compliant_uri(artifact_uri)
        super().__init__(
            artifact_uri,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            credential_refresh_def=credential_refresh_def,
            addressing_style="virtual",
            s3_endpoint_url=s3_endpoint_url,
            s3_upload_extra_args=s3_upload_extra_args,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )

    # Cloudflare implementation of head_bucket is not the same as AWS's, so we
    # temporarily use the old method of get_bucket_location until cloudflare
    # updates their implementation
    def _get_region_name(self):
        # note: s3 client enforces path addressing style for get_bucket_location
        temp_client = _get_s3_client(
            addressing_style="path",
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            session_token=self._session_token,
            s3_endpoint_url=self._s3_endpoint_url,
        )
        return temp_client.get_bucket_location(Bucket=self.bucket)["LocationConstraint"]

    def parse_s3_compliant_uri(self, uri):
        # r2 uri format(virtual): r2://<bucket-name>@<account-id>.r2.cloudflarestorage.com/<path>
        parsed = urlparse(uri)
        if parsed.scheme != "r2":
            raise Exception(f"Not an R2 URI: {uri}")

        host = parsed.netloc
        path = parsed.path

        bucket = host.split("@")[0]
        path = path.removeprefix("/")
        return bucket, path

    @staticmethod
    def convert_r2_uri_to_s3_endpoint_url(r2_uri):
        host = urlparse(r2_uri).netloc
        host_without_bucket = host.split("@")[-1]
        return f"https://{host_without_bucket}"
